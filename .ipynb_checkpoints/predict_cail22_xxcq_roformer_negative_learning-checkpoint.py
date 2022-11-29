import torch
import json
import sys
import numpy as np
import torch.nn as nn
from albert_models.nets.gpNet import RawGlobalPointer
from transformers import BertTokenizerFast, BertModel
from torch.utils.data import DataLoader
import configparser
import logging
from tqdm import tqdm
from torch.utils.data.dataset import ConcatDataset

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

import os
cur_dir_path = os.path.dirname(__file__)
sys.path.extend([cur_dir_path])

# # 控制台参数传入
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
        import os, sys, torch

        con = configparser.ConfigParser()
        con_path = os.path.join(cur_dir_path, args['config_file'])
        con.read(con_path, encoding='utf8')

        print(cur_dir_path, '======', con_path)

        args_path = dict(dict(con.items('paths')), **dict(con.items("para")))
        # try:
        #     tokenizer = BertTokenizerFast.from_pretrained(args_path["model_path"], do_lower_case=True)
        # except:
        tokenizer = BertTokenizerFast.from_pretrained(args_path["vocab_path"], do_lower_case=True)

        tokenizer.add_special_tokens({
                            "additional_special_tokens": ['[CLS]', '[SEP]']})

        print(args_path["model_path"], '=====')

        from albert_models.modeling_roformer import RoFormerModel
        from albert_models.configuration_roformer import RoFormerConfig
        # from roformer import RoFormerModel, RoFormerConfig
        config = RoFormerConfig.from_pretrained(args_path["model_path"])
        encoder = RoFormerModel(config=config)

        idx = 0
        schema = {}
        link_symbol = args_path['link_symbol']
        for schema_path in args_path["predict_schema_data"].split(','):
            schema_path = os.path.join(cur_dir_path, schema_path)
            with open(schema_path, 'r', encoding='utf-8') as f:
                for _, item in enumerate(f):
                    item = json.loads(item.rstrip())
                    for key in item['object_type']:
                        schema[item["subject_type"]+link_symbol+item["predicate"]+link_symbol+item['object_type'][key]] = idx
                        idx += 1

        print(schema)

        id2schema = {}
        for k,v in schema.items(): id2schema[v]=k

        device = torch.device("cuda:0")

        mention_detect = RawGlobalPointer(hiddensize=config.hidden_size, ent_type_size=2, inner_dim=con.getint("para", "head_size"), RoPE=True).to(device)#实体关系抽取任务默认不提取实体类型
        s_o_head = RawGlobalPointer(hiddensize=config.hidden_size, ent_type_size=1, inner_dim=con.getint("para", "head_size"), RoPE=False, tril_mask=False).to(device)
        s_o_tail = RawGlobalPointer(hiddensize=config.hidden_size, ent_type_size=1, inner_dim=con.getint("para", "head_size"), RoPE=False, tril_mask=False).to(device)

        class ERENet(nn.Module):
            def __init__(self, encoder, a, b, c):
                super(ERENet, self).__init__()
                self.mention_detect = a
                self.s_o_head = b
                self.s_o_tail = c
                self.encoder = encoder

            def forward(self, batch_token_ids, batch_mask_ids, batch_token_type_ids):
                outputs = self.encoder(batch_token_ids, batch_mask_ids, batch_token_type_ids)

                mention_outputs = self.mention_detect(outputs, batch_mask_ids)
                so_head_outputs = self.s_o_head(outputs, batch_mask_ids)
                so_tail_outputs = self.s_o_tail(outputs, batch_mask_ids)
                return mention_outputs, so_head_outputs, so_tail_outputs

        net = ERENet(encoder, mention_detect, s_o_head, s_o_tail).to(device)

        eo = 19
        output_path = os.path.join(cur_dir_path, args_path['output_path'])
        print(output_path, '==output_path==')
        import os
        try:
            ckpt = torch.load(os.path.join(output_path, 'spo_conv_asa.pth.{}.fp16'.format(eo)))
            net.load_state_dict(ckpt)
            print('===succeeded in loading fp16===')
        except:
            ckpt = torch.load(os.path.join(output_path, 'spo_conv_asa.pth.{}.fp16'.format(eo)))
            new_ckpt = {}
            for key in ckpt:
                name = key.split('.')
                new_ckpt[".".join(name[1:])] = ckpt[key]
            net.load_state_dict(new_ckpt)
            print('===succeeded in loading fp16===')

        import torch
        """
        https://github.com/suvojit-0x55aa/mixed-precision-pytorch/blob/master/train.py
        """
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
        #     torch.save(net.state_dict(), os.path.join(output_path, 'spo_conv_asa.pth.{}.fp16'.format(eo)))

        net.eval()

        threshold_so_prob = 0.49
        threshold_so = np.log(threshold_so_prob/(1-threshold_so_prob))

        threshold_pair_prob = 0.49
        threshold_pair = np.log(threshold_pair_prob/(1-threshold_pair_prob))

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

        def extract_spoes(text, threshold_so=0, threshold_pair=0):
            """抽取输入text所包含的三元组
            """
            encoder_text = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
            token2char_span_mapping = encoder_text['offset_mapping']
            new_span, entities = [], []
            for i in token2char_span_mapping:
                if i[0] == i[1]:
                    new_span.append([])
                else:
                    if i[0] + 1 == i[1]:
                        new_span.append([i[0]])
                    else:
                        new_span.append([i[0], i[-1] - 1])

            input_ids = encoder_text["input_ids"]
            doc_spans = slide_window(input_ids, con.getint("para", "maxlen"), 32, offset=8)
            spo_list = []

            for doc_span in doc_spans:
                span_start = doc_span.start
                span_end = doc_span.start + doc_span.length

                for candidate_type in schema:
                    candidate_start = 0
                    candidate_type_ids = tokenizer(candidate_type, add_special_tokens=False)['input_ids']

                    # [cls]candidate_type[sep]
                    span_input_ids = tokenizer('[CLS]', add_special_tokens=False)['input_ids'] + candidate_type_ids + tokenizer('[SEP]', add_special_tokens=False)['input_ids']
                    candidate_start = len(span_input_ids)

                    span_input_ids += input_ids[span_start:span_end] + tokenizer('[SEP]', add_special_tokens=False)['input_ids']
                    span_type_ids = [0] * len(span_input_ids)
                    span_attention_mask = [1] * len(span_input_ids)

                    span_input_ids = torch.tensor(span_input_ids).long().unsqueeze(0).to(device)
                    span_type_ids = torch.tensor(span_type_ids).unsqueeze(0).to(device)
                    span_attention_mask = torch.tensor(span_attention_mask).unsqueeze(0).to(device)

                    with torch.no_grad():
                        scores = net(span_input_ids, span_attention_mask, span_type_ids)
                        sigmoid_scores = [torch.nn.Sigmoid()(score) for score in scores]

                    outputs = [o[0].data.cpu().numpy() for o in scores]
                    outputs_scores = [o[0].data.cpu().numpy() for o in sigmoid_scores]

                    subjects, objects = set(), set()
                    outputs[0][:, [0, -1]] -= np.inf
                    outputs[0][:, :, [0, -1]] -= np.inf

                    for l, h, t in zip(*np.where(outputs[0] > threshold_so)):
                        if h < candidate_start and t < candidate_start:
                            continue
                        if l == 0:
                            subjects.add((h, t, outputs_scores[0][0, h, t]))
                        else:
                            objects.add((h, t, outputs_scores[0][1, h, t]))
                    spoes = set()
                    for sh, st, s_score in subjects:
                        for oh, ot, o_score in objects:
                            p1s = np.where(outputs[1][:, sh, oh] > threshold_pair)[0]
                            p2s = np.where(outputs[2][:, st, ot] > threshold_pair)[0]
                            ps = set(p1s) & set(p2s)
                            for p in ps:

                                p1_score = (outputs_scores[1][p, sh, oh])
                                p2_score = (outputs_scores[2][p, st, ot])
                                p_score = (p1_score+p2_score)/2.0

                                tmp_score = (s_score, o_score, p_score)

                                sh_span = sh + span_start - candidate_start
                                st_span = st + span_start - candidate_start
                                oh_span = oh + span_start - candidate_start
                                ot_span = ot + span_start - candidate_start

                                try:
                                    spoes.add((candidate_type, 
                                        text[new_span[sh_span][0]:new_span[st_span][-1] + 1], 
                                        new_span[sh_span][0], new_span[st_span][-1] + 1, 
                                        text[new_span[oh_span][0]:new_span[ot_span][-1] + 1], 
                                        new_span[oh_span][0], new_span[ot_span][-1] + 1, 
                                        tmp_score
                                    ))
                                except:
                                    continue
                    spoes = list(spoes)
                    for spo in spoes:
                        """
                        (spo["subject"], spo["predicate"], spo["object"][key], spo["subject_type"], spo["object_type"][key])
                        """
                        [subject_type, predicate, object_type] = spo[0].split(link_symbol)

                        spo_list.append({
                            'label':inverse_mapping[predicate],
                            'em1Text': spo[1],
                            'em2Text': spo[4],
                            'em1Text_pos': [int(spo[2]), int(spo[3])],
                            'em2Text_pos': [int(spo[5]), int(spo[6])],
                            'em1Text_score': float(spo[-1][0]),
                            'em2Text_score': float(spo[-1][1]),
                            'label_score': float(spo[-1][-1])
                        })
            return spo_list

        cnt = 0
        from tqdm import tqdm
        import os

        output_file = os.path.join(cur_dir_path, args['output_file'])
        print(output_file, '===output_file===')

        with open(output_file, 'w') as fwobj:
            with open(args['input_file'], 'r') as frobj:
                for idx, line in tqdm(enumerate(frobj)):
                    tmp = json.loads(line.strip())
                    content = {
                        'text':tmp['sentText']
                    }
                    tmp_dict = {
                        'entityMentions': [],
                        'relationMentions': [],
                        'articleId': str(idx),
                        'sentID': str(idx+100000),
                        'sentText': content['text']
                    }
                    spo_list = extract_spoes(content['text'], threshold_so=threshold_so, threshold_pair=threshold_pair)
                    tmp_dict['relationMentions'] = spo_list
                    fwobj.write(json.dumps(tmp_dict, ensure_ascii=False)+'\n')
            