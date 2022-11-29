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
from transformers import BertTokenizerFast
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
from torch.nn.modules.loss import _Loss
from tqdm import tqdm
from torch.utils.data.dataset import ConcatDataset

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

import os
cur_dir_path = os.path.dirname(__file__)
sys.path.extend([cur_dir_path])

# import argparse
# def parse_args():
#     parser = argparse.ArgumentParser(description=__doc__)
#     parser.add_argument(
#         "--input_file",
#         default=None,
#         type=str,
#         help="The config file.", )
    
#     parser.add_argument(
#         "--config_file",
#         default=None,
#         type=str,
#         help="The config file.", )
    
#     parser.add_argument(
#         "--output_file",
#         default=None,
#         type=str,
#         help="The out tag added to output_path.", )

#     args = parser.parse_args()

#     return args

# args = parse_args() # 从命令行获取

class Predict(object):
    def __init__(self, args):
        
        import os, sys
        import torch

        con = configparser.ConfigParser()
        con_path = os.path.join(cur_dir_path, args['config_file'])
        con.read(con_path, encoding='utf8')

        args_path = dict(dict(con.items('paths')), **dict(con.items("para")))
        # tokenizer = BertTokenizerFast.from_pretrained('hfl/chinese-roberta-wwm-ext', do_lower_case=True)
        
        vocab_path = os.path.join(cur_dir_path, args_path['vocab_path'])
        tokenizer = BertTokenizerFast.from_pretrained(vocab_path, do_lower_case=True)

        print(tokenizer.tokenize('我是中国人[SEP]'))

        class MyUniLM(nn.Module):
            def __init__(self, config_path, model_path, eos_token_id, **kargs):
                super().__init__()

                self.model_path = model_path
                self.config_path = config_path
                self.eos_token_id = eos_token_id

                if args_path['model_type'] == 'roberta':

                    from albert_models.nets.unilm_bert import BertForCausalLM
                    from transformers import BertConfig

                    self.config = BertConfig.from_pretrained(config_path)
                    self.config.is_decoder = True
                    self.config.eos_token_id = self.eos_token_id

                    self.transformer = BertForCausalLM(
                                            config=self.config)

                elif args_path['model_type'] == 'roformer':
                    from albert_models.nets.unilm_roformer import RoFormerForCausalLM
                    # from roformer import RoFormerConfig
                    from albert_models.configuration_roformer import RoFormerConfig

                    self.config = RoFormerConfig.from_pretrained(config_path)
                    self.config.is_decoder = True
                    self.config.eos_token_id = self.eos_token_id

                    self.transformer = RoFormerForCausalLM(
                                            config=self.config)

            def forward(self, input_ids, input_mask, segment_ids=None, mode='train', **kargs):
                if mode == "train":
                    idxs = torch.cumsum(segment_ids, dim=1)
                    attention_mask_3d = (idxs[:, None, :] <= idxs[:, :, None]).to(dtype=torch.float32)
                    model_outputs = self.transformer(input_ids, 
                                                     attention_mask=attention_mask_3d, 
                                                     token_type_ids=segment_ids)
                    return model_outputs # return prediction-scores
                elif mode == "generation":
                    model_outputs = self.transformer.generate(
                                                    input_ids=input_ids, 
                                                    attention_mask=input_mask, 
                                                    token_type_ids=segment_ids, 
                                                    **kargs) # we need to generate output-scors
                return model_outputs

        duie_task_dict = {
            'sep_token':'[SEP]',
            'seg_token':'<S>',
            'group_token':'<T>',
            'start_token':'[CLS]',
            'end_token':'[SEP]',
            'sentinel_token':'[unused{}]',
            'instruction':'信息抽取',
            'sentinel_start_idx':1,
            'add_schema_type':True
        }

        duee_task_dict = {
            'sep_token':'[SEP]',
            'seg_token':'<S>',
            'group_token':'<T>',
            'start_token':'[CLS]',
            'end_token':'[SEP]',
            'sentinel_token':'[unused{}]',
            'instruction':'事件抽取',
            'sentinel_start_idx':1,
            'add_schema_type':True
        }

        entity_task_dict = {
            'sep_token':'[SEP]',
            'seg_token':'<S>',
            'group_token':'<T>',
            'start_token':'[CLS]',
            'end_token':'[SEP]',
            'sentinel_token':'[unused{}]',
            'instruction':'实体抽取',
            'sentinel_start_idx':1,
            'add_schema_type':False
        }

        schema = []
        schema_path_dict = {}
        for schema_info in args_path["schema_data"].split(','):
            schema_type, schema_path = schema_info.split(':')
            schema_path = os.path.join(cur_dir_path, schema_path)
            schema_tuple = tuple(schema_path.split('/')[:-1])
            if schema_type not in schema_path_dict:
                schema_path_dict[schema_type] = []
            print(schema_type, schema_path, '===schema-path===', schema_type)
            if 'duie' in schema_type:
                schema.extend(load_ie_schema(schema_path))
                schema_path_dict[schema_type] = load_ie_schema(schema_path)
            elif 'duee' in schema_type:
                schema.extend(load_ee_schema(schema_path))
                schema_path_dict[schema_type] = load_ee_schema(schema_path)
            elif 'entity' in schema_type:
                schema.extend(load_entity_schema(schema_path))
                schema_path_dict[schema_type] = load_entity_schema(schema_path)

        print(schema_path_dict, '==schema_path_dict==')

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
            instruction_token_type_ids = encoder_instruction_text["token_type_ids"] #RoBERTa不需要NSP任务
            instruction_attention_mask = encoder_instruction_text["attention_mask"] #RoBERTa不需要NSP任务

            # generate input-ids
            encoder_text = tokenizer(text, return_offsets_mapping=True, truncation=False, add_special_tokens=False)
            input_ids = encoder_text["input_ids"]
            token_type_ids = encoder_text["token_type_ids"] #RoBERTa不需要NSP任务
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
            schema_token_type_ids = encoder_schema_text["token_type_ids"] #RoBERTa不需要NSP任务
            schema_attention_mask = encoder_schema_text["attention_mask"]

            output_list = []
            doc_spans = slide_window(input_ids, max_len, 64, offset=0)
            for doc_span in doc_spans:
                span_start = doc_span.start
                span_end = doc_span.start + doc_span.length

                span_input_ids = input_ids[span_start:span_end] + tokenizer(task_dict['sep_token'], add_special_tokens=False)['input_ids']

                span_type_ids = [0] * len(span_input_ids)
                span_attention_mask = [1] * len(span_input_ids)
                output_list.append((offset_mapping, instruction_input_ids, span_input_ids, schema_input_ids, input_ids))
            return output_list

        output_path = os.path.join(cur_dir_path, args_path['output_path'])
        config_path = os.path.join(cur_dir_path, args_path['config_path'])

        device = torch.device("cuda:0")
        net = MyUniLM(config_path=config_path, 
                      model_path='', 
                      eos_token_id=tokenizer.sep_token_id)
        net.to(device)
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

        print(output_path, '=====')

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

        # if con.getboolean('para', 'fp_16'):
        #     net = net.half()
        #     print("===do fp16 inference===")
        #     # torch.save(net.state_dict(), os.path.join(output_path, 'unilm_mixture.pth.{}.fp16'.format(eo)))

        net = net.half()
        def predict_seq2struct(net, decoder, data, task_dict, schema_dict):
            decoded_list = []
            for target_type in data['schema_type']:            
                # [('丈夫', '人物', '赵楚'), ('丈夫', '人物', '柳思孝')]
                output_list = extract(data, task_dict, schema_dict, target_type, max_len=con.getint("para", "maxlen"))
                for output_input in output_list:
                    (offset_mapping, instruction_input_ids, input_ids, schema_input_ids, ori_input_ids) = output_input
                    text = data['text']

                    query_token_ids = instruction_input_ids + input_ids + schema_input_ids

                    from albert_models.nets.constrained_decoder import get_end_to_end_prefix_allowed_tokens_fn_hf
                    prefix_allowed_tokens_fn = get_end_to_end_prefix_allowed_tokens_fn_hf(input_ids, task_dict, tokenizer)

                    batch_token_ids = torch.tensor(query_token_ids).long().unsqueeze(0).to(device)
                    batch_mask_ids = torch.tensor([1] * len(query_token_ids)).long().unsqueeze(0).to(device)
                    batch_token_type_ids = torch.tensor([0] * len(query_token_ids)).long().unsqueeze(0).to(device)

                    with torch.no_grad():
                        model_outputs = net(input_ids=batch_token_ids, input_mask=batch_mask_ids, 
                                            segment_ids=batch_token_type_ids, mode='generation',
                           output_scores=True, do_sample=False, max_length=1024, num_beams=2, 
                            return_dict_in_generate=True, prefix_allowed_tokens_fn=prefix_allowed_tokens_fn
                           )

        #             batch_token_ids = torch.tensor(query_token_ids).long().unsqueeze(0).to(device)
        #             batch_mask_ids = torch.tensor([1] * len(query_token_ids)).long().unsqueeze(0).to(device)
        #             batch_token_type_ids = torch.tensor([0] * len(query_token_ids)).long().unsqueeze(0).to(device)

        #             model_outputs = net(input_ids=batch_token_ids, input_mask=batch_mask_ids, segment_ids=batch_token_type_ids, mode='generation',
        #                    output_scores=True, do_sample=False, max_length=1024, num_beams=2, return_dict_in_generate=True)

                    decode_output_list = decoder.single_schema_decode(text, offset_mapping, target_type, model_outputs, query_token_ids, ori_input_ids, mode='unilm')

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
        print(output_file, '===output_file===')

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