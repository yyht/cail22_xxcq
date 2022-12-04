# -*- coding: utf-8 -*-
"""
@Auth: Xhw
@Description: token-pair范式的实体关系抽取pytorch实现
"""
import torch
import json
import sys, os
import numpy as np
import torch.nn as nn
from transformers import T5TokenizerFast
from albert_models.utils.seq2struct_dataloader import (
                                         load_ie_schema, load_ee_schema, load_entity_schema, 
                                         load_entity, load_duie, load_duee)
from torch.utils.data import DataLoader
import configparser
import logging
import torch
import io
import torch.nn.functional as F
import random
import numpy as np
import time
import math
import datetime
import torch.nn as nn
import logging
from tqdm import tqdm

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

import os
cur_dir_path = os.path.dirname(__file__)
sys.path.extend([cur_dir_path])

class Predict(object):
    def __init__(self, args):
        
        import os, sys, torch
        
        con = configparser.ConfigParser()
        con_path = os.path.join(cur_dir_path, args['config_file'])
        con.read(con_path, encoding='utf8')

        args_path = dict(dict(con.items('paths')), **dict(con.items("para")))
        tokenizer = T5TokenizerFast.from_pretrained(args_path['vocab_path'])

        class MyT5LM(nn.Module):
            def __init__(self, config_path, model_path, **kargs):
                super().__init__()
                from transformers import T5ForConditionalGeneration, T5Config
                self.config = T5Config.from_pretrained(config_path)
                self.transformer = T5ForConditionalGeneration(config=self.config)

            def forward(self, input_ids, input_mask, decoder_input_ids, decoder_input_mask, mode='train', **kargs):
                if mode == "train":
                    model_outputs = self.transformer(input_ids=input_ids, 
                                                     attention_mask=input_mask,
                                                    decoder_input_ids=decoder_input_ids,
                                                    decoder_attention_mask=decoder_input_mask)
                    return model_outputs # return prediction-scores
                elif mode == "generation":
                    model_outputs = self.transformer.generate(
                                                    input_ids=input_ids, 
                                                    attention_mask=input_mask,
                                                    # decoder_input_ids=decoder_input_ids,
                                                    # decoder_attention_mask=decoder_input_mask,
                                                    **kargs) # we need to generate output-scors
                return model_outputs

        duie_task_dict = {
            'sep_token':'<extra_id_1>',
            'seg_token':'<extra_id_2>',
            'group_token':'<extra_id_3>',
            'start_token':'',
            'end_token':'</s>',
            'sentinel_token':'<extra_id_{}>',
            'instruction':'信息抽取',
            'sentinel_start_idx':20,
            'add_schema_type':True,
            "greedy_search":True,
            'role_shuffle':True,
            'role_schema_order': True,
            'remove_dup':True,
            'search_mode':'string'
        }

        duee_task_dict = {
            'sep_token':'<extra_id_1>',
            'seg_token':'<extra_id_2>',
            'group_token':'<extra_id_3>',
            'start_token':'',
            'end_token':'</s>',
            'sentinel_token':'<extra_id_{}>',
            'instruction':'事件抽取',
            'sentinel_start_idx':20,
            'add_schema_type':True,
            "greedy_search":False,
            'role_shuffle':True,
            'role_schema_order': True,
            'remove_dup':True,
            'search_mode':'string'
        }

        entity_task_dict = {
            'sep_token':'<extra_id_1>',
            'seg_token':'<extra_id_2>',
            'group_token':'<extra_id_3>',
            'start_token':'',
            'end_token':'</s>',
            'sentinel_token':'<extra_id_{}>',
            'instruction':'实体抽取',
            'sentinel_start_idx':20,
            'add_schema_type':False,
            "greedy_search":False,
            'role_shuffle':True,
            'remove_dup':True,
            'search_mode':'string'
        }

        schema = []
        schema_path_dict = {}
        for schema_info in args_path["schema_data"].split(','):
            schema_type, schema_path = schema_info.split(':')
            schema_path = os.path.join(cur_dir_path, schema_path)
            schema_tuple = tuple(schema_path.split('/')[:-1])
            if schema_type not in schema_path_dict:
                schema_path_dict[schema_type] = []
            # print(schema_type, schema_path, '===schema-path===', schema_type)
            if 'duie' in schema_type:
                schema.extend(load_ie_schema(schema_path))
                schema_path_dict[schema_type] = load_ie_schema(schema_path)
            elif 'duee' in schema_type:
                schema.extend(load_ee_schema(schema_path))
                schema_path_dict[schema_type] = load_ee_schema(schema_path)
            elif 'entity' in schema_type:
                schema.extend(load_entity_schema(schema_path))
                schema_path_dict[schema_type] = load_entity_schema(schema_path)

        # print(schema_path_dict, '==schema_path_dict==')

        def search(pattern, sequence):
            """从sequence中寻找子串pattern
            如果找到，返回第一个下标；否则返回-1。
            """
            n = len(pattern)
            for i in range(len(sequence)):
                if sequence[i:i + n] == pattern:
                    return i
            return -1

        from collections import namedtuple
        _DocSpan = namedtuple(  # pylint: disable=invalid-name
                "DocSpan", ["start", "length"])

        def slide_window(all_doc_tokens, max_length, doc_stride, offset=32):
            doc_spans = []
            start_offset = 0
            while start_offset < len(all_doc_tokens):
                length = len(all_doc_tokens) - start_offset
                if length > max_length - offset:
                    length = max_length - offset
                doc_spans.append(_DocSpan(start=start_offset, length=length))
                if start_offset + length == len(all_doc_tokens):
                    break
                start_offset += min(length, doc_stride)
            return doc_spans

        def extract(item, task_dict, all_schema_dict, target_type, max_len):
            text = item['text']

            # generate instruction input-ids
            instruction_text = task_dict['start_token'] + task_dict['instruction'] + task_dict['sep_token']
            encoder_instruction_text = tokenizer(instruction_text, return_offsets_mapping=True, add_special_tokens=False)
            instruction_input_ids = encoder_instruction_text["input_ids"]
            instruction_token_type_ids = encoder_instruction_text.get("token_type_ids", []) #RoBERTa不需要NSP任务
            instruction_attention_mask = encoder_instruction_text["attention_mask"] #RoBERTa不需要NSP任务

            # generate input-ids
            encoder_text = tokenizer(text, return_offsets_mapping=True, truncation=False, add_special_tokens=False)
            input_ids = encoder_text["input_ids"]
            token_type_ids = encoder_text.get("token_type_ids", []) #RoBERTa不需要NSP任务
            attention_mask = encoder_text["attention_mask"]

            # generate schema
            offset_mapping = encoder_text.offset_mapping
            schema_dict = all_schema_dict[target_type]
            if task_dict['add_schema_type']:
                schema_strings = target_type + task_dict['sep_token']
            else:
                schema_strings = ''
            for role in schema_dict['role2sentinel']:
                schema_strings += role + schema_dict['role2sentinel'][role] # role sentinel

            schema_strings += task_dict['sep_token']

            encoder_schema_text = tokenizer(schema_strings, return_offsets_mapping=True, max_length=max_len, truncation=False, add_special_tokens=False)
            schema_input_ids = encoder_schema_text["input_ids"]
            schema_token_type_ids = encoder_schema_text.get("token_type_ids", []) #RoBERTa不需要NSP任务
            schema_attention_mask = encoder_schema_text["attention_mask"]

            output_list = []
            doc_spans = slide_window(input_ids, max_len, 64, offset=0)
            for doc_span in doc_spans:
                span_start = doc_span.start
                span_end = doc_span.start + doc_span.length

                span_input_ids = input_ids[span_start:span_end] + tokenizer(task_dict['end_token'], add_special_tokens=False)['input_ids']

                span_type_ids = [0] * len(span_input_ids)
                span_attention_mask = [1] * len(span_input_ids)
                output_list.append((offset_mapping, instruction_input_ids, span_input_ids, schema_input_ids, input_ids))
            return output_list

        output_path = os.path.join(cur_dir_path, args_path['output_path'])
        config_path = os.path.join(cur_dir_path, args_path['config_path'])

        # print(tokenizer.eos_token_id)

        device = torch.device("cuda:0")
        net = MyT5LM(config_path=args_path["config_path"],
                    model_path=args_path["output_path"])
        net.to(device)
        # eo = 9
        eo = 19
        try:
            ckpt = torch.load(os.path.join(output_path, 'unilm_mixture.pth.{}.fp16'.format(eo)))
            net.load_state_dict(ckpt)
        except:
            ckpt = torch.load(os.path.join(output_path, 'unilm_mixture.pth.{}.fp16'.format(eo)))
            new_ckpt = {}
            for key in ckpt:
                name = key.split('.')
                new_ckpt[".".join(name[1:])] = ckpt[key]
            net.load_state_dict(new_ckpt)
        net.eval()

        net = net.half()

        import torch
        class tofp16(nn.Module):
            """
            Add a layer so inputs get converted to FP16.
            Model wrapper that implements::
                def forward(self, input):
                    return input.half()
            """

            def __init__(self):
                super().__init__()

            def forward(self, input):
                return input.half()

        def predict_seq2struct(net, decoder, data, task_dict, schema_dict):
            decoded_list = []
            for target_type in data['schema_type']:            
                # [('丈夫', '人物', '赵楚'), ('丈夫', '人物', '柳思孝')]
                output_list = extract(data, task_dict, schema_dict, target_type, max_len=con.getint("para", "maxlen"))
                for output_input in output_list:
                    (offset_mapping, instruction_input_ids, input_ids, schema_input_ids, ori_input_ids) = output_input
                    text = data['text']

                    query_token_ids = instruction_input_ids + input_ids + schema_input_ids

                    # from nets.constrained_decoder import get_end_to_end_prefix_allowed_tokens_fn_hf
                    # prefix_allowed_tokens_fn = get_end_to_end_prefix_allowed_tokens_fn_hf(input_ids, task_dict, tokenizer)

                    batch_token_ids = torch.tensor(query_token_ids).long().unsqueeze(0).to(device)
                    batch_mask_ids = torch.tensor([1] * len(query_token_ids)).long().unsqueeze(0).to(device)
                    # batch_token_type_ids = torch.tensor([0] * len(query_token_ids)).long().unsqueeze(0).to(device)

                    with torch.no_grad():
                        model_outputs = net(input_ids=batch_token_ids, input_mask=batch_mask_ids, decoder_input_ids=None, 
                            decoder_input_mask=None, mode='generation',
                            output_scores=True, do_sample=False, max_length=1024, num_beams=2, return_dict_in_generate=True,
                            # prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                           )

        #             batch_token_ids = torch.tensor(query_token_ids).long().unsqueeze(0).to(device)
        #             batch_mask_ids = torch.tensor([1] * len(query_token_ids)).long().unsqueeze(0).to(device)
        #             batch_token_type_ids = torch.tensor([0] * len(query_token_ids)).long().unsqueeze(0).to(device)

        #             model_outputs = net(input_ids=batch_token_ids, input_mask=batch_mask_ids, segment_ids=batch_token_type_ids, mode='generation',
        #                    output_scores=True, do_sample=False, max_length=1024, num_beams=2, return_dict_in_generate=True)

                    decode_output_list = decoder.single_schema_decode(text, offset_mapping, target_type, model_outputs, query_token_ids, ori_input_ids,
                                                                      mode='seq2seq_no_decoder_input', search_mode='string')

                    for output in decode_output_list:
                        # [('正向情感', '意见对象', '阿婴'), ('正向情感', '情感表达', '挺刺激')]
                        # if len(output) >= 2:
                        decoded_list.append(output)
            return decoded_list

        def build_schema(task_dict, schema_dict_list):
            sentinel_token = task_dict['sentinel_token']
            sentinel_start_idx = task_dict['sentinel_start_idx']

            all_schema_dict = {}
            for schema_dict in schema_dict_list:
                if schema_dict['type'] not in all_schema_dict:
                    all_schema_dict[schema_dict['type']] = {
                            'role2sentinel':{},
                            'sentinel2role':{},
                            'role2type':{},
                            'type2role':{},
                            'role_index': 0
                    }
                role_index = all_schema_dict[schema_dict['type']]['role_index']
                for _, role_dict in enumerate(schema_dict['role_list']):
                    role_type = role_dict['type'] + role_dict['role']
                    if role_type in all_schema_dict[schema_dict['type']]['role2sentinel']:
                        continue
                    all_schema_dict[schema_dict['type']]['role2sentinel'][role_type] = sentinel_token.format(role_index+sentinel_start_idx)
                    all_schema_dict[schema_dict['type']]['sentinel2role'][sentinel_token.format(role_index+sentinel_start_idx)] = role_type
                    all_schema_dict[schema_dict['type']]['role2type'][role_dict['role']] = role_type
                    all_schema_dict[schema_dict['type']]['type2role'][role_type] = role_dict['role']
                    role_index += 1
                all_schema_dict[schema_dict['type']]['role_index'] = role_index
            from albert_models.utils.seq2struct_decoder import single_schema_decoder as single_schema_decoder
            decoder = single_schema_decoder(tokenizer, con.getint("para", "maxlen"), schema_dict_list, label=0, 
                         task_dict=task_dict, mode='train')
            return all_schema_dict, decoder

        mapping = {
            'Ns': '地理位置',
            'Nh': '人物',
            'NT': '时间',
            'NDR': '毒品',
            'NW': '重量',
            'sell_drugs_to': '贩卖毒品给人',
            'traffic_in': '贩卖毒品',
            'posess': '持有毒品',
            'provide_shelter_for': '非法容留摄入或注射某毒品'
        }

        inverse_mapping = {}
        for key in mapping:
            inverse_mapping[mapping[key]] = key

        from itertools import combinations
        from tqdm import tqdm
        import re

        def search_all(pattern, sequence):
            """从sequence中寻找子串pattern
            如果找到，返回第一个下标；否则返回-1。
            """
            n = len(pattern)
            h = []
            for i in range(len(sequence)):
                if sequence[i:i + n] == pattern:
                    h.append(i)
            return h

        total_dict = {}

        import os
        import os

        from itertools import product
        import numpy as np

        output_file = os.path.join(cur_dir_path, args['output_file'])

        with open(output_file, 'w') as fwobj:
            with open(args['input_file'], 'r') as frobj:
                for idx, line in tqdm(enumerate(frobj)):
                    content = {
                        'text':line.strip()
                    }
                    tmp_dict = {
                        'entityMentions': [],
                        'relationMentions': [],
                        'articleId': str(idx),
                        'sentID': str(idx+100000),
                        'sentText': content['text']
                    }

                    all_schema_dict, decoder = build_schema(duie_task_dict, schema_path_dict['duie_cail22_xxcq'])
                    content['schema_type'] = list(all_schema_dict.keys())
                    decoded_list = predict_seq2struct(net, decoder, content, duie_task_dict, all_schema_dict)
                    for decoded in decoded_list: 
                        """
                        {'e1start': 17,
                       'em2Text': '海洛因',
                       'e21start': 48,
                       'label': 'traffic_in',
                       'em1Text': '林某某'}
                        """
                        if len(decoded) == 2:
                            em1Text = decoded[0][2]
                            em2Text = decoded[1][2]
                            predicate_label = inverse_mapping[decoded[0][0]]
                            tmp_dict['relationMentions'].append({
                                'label':predicate_label,
                                'em1Text': em1Text,
                                'em2Text': em2Text
                            })
                    fwobj.write(json.dumps(tmp_dict, ensure_ascii=False)+'\n')