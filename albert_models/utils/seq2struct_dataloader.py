# -*- coding: utf-8 -*-
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from operator import is_not
from functools import partial
import random
from torch.utils.data import Dataset, DataLoader, BatchSampler, Sampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from collections import OrderedDict

"""
@Auth: Xhw
@Description: CHIP/CBLUE 医学实体关系抽取，数据来源 https://tianchi.aliyun.com/dataset/dataDetail?dataId=95414
"""
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import random

def load_mrc_schema(filename):
    D = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            schema_dict = {
                'type':line,
                'role_list':[{'role':line, 'type':''}]
            }
            D.append(schema_dict)
    return D

def load_entity_schema(filename):
    """
    {"entity_type": "地理位置"}
    to 
    {"type": "裁员", "role_list": [{"role": "裁员方"}, {"role": "裁员人数"}, {"role": "时间"}]}
    """
    D = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = json.loads(line.strip())
            schema_dict = {
                'type':line['entity_type'],
                'role_list':[{'role':line['entity_type'], 'type':''}]
            }
            D.append(schema_dict)
    print(D)
    return D
    
def load_ie_schema(filename):
    """
    {"predicate": "父亲", "subject_type": "人物", "object_type": {"a1": "人物"}}
    to
    {"type": "裁员", "role_list": [{"role": "裁员方"}, {"role": "裁员人数"}, {"role": "时间"}]}
    """
    D = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = json.loads(line.strip())
            schema_dict = {
                'type':line['predicate'],
                'role_list':[{'role':line['subject_type'], 'type':'subject-'}]
            }
            for key in line['object_type']:
                schema_dict['role_list'].append({'role':line['object_type'][key], 'type':'object-'})
            D.append(schema_dict)
    return D

def load_ee_schema(filename):
    """
    {"predicate": "父亲", "subject_type": "人物", "object_type": {"a1": "人物"}}
    to
    {"type": "裁员", "role_list": [{"role": "裁员方"}, {"role": "裁员人数"}, {"role": "时间"}]}
    """
    D = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = json.loads(line.strip())
            schema_dict = {
                'type':line['event_type'],
                'role_list':[{'role':'触发词', 'type':''}]
            }
            for role_dict in line['role_list']:
                role_dict['type'] = ''
                schema_dict['role_list'].append(role_dict)
            D.append(schema_dict)
    return D

def load_ee_no_trigger_schema(filename):
    """
    {"predicate": "父亲", "subject_type": "人物", "object_type": {"a1": "人物"}}
    to
    {"type": "裁员", "role_list": [{"role": "裁员方"}, {"role": "裁员人数"}, {"role": "时间"}]}
    """
    D = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = json.loads(line.strip())
            schema_dict = {
                'type':line['event_type'],
                'role_list':[]
            }
            for role_dict in line['role_list']:
                role_dict['type'] = ''
                schema_dict['role_list'].append(role_dict)
            D.append(schema_dict)
    return D

def load_entity(filename):
    """
    {"text": "对儿童SARST细胞亚群的研究表明，与成人SARS相比，儿童细胞下降不明显，证明上述推测成立。", "entity_list": [{"start_idx": 3, "end_idx": 9, "type": "身体部位", "entity": "SARST细胞"}, {"start_idx": 19, "end_idx": 24, "type": "疾病", "entity": "成人SARS"}]}
    to
     {"text": "消失的“外企光环”，5月份在华裁员900余人，香饽饽变“臭”了", "id": "cba11b5059495e635b4f95e7484b2684", "event_list": [{"event_type": "组织关系-裁员", "trigger": "裁员", "trigger_start_index": 15, "arguments": [{"argument_start_index": 17, "role": "裁员人数", "argument": "900余人", "alias": []}, {"argument_start_index": 10, "role": "时间", "argument": "5月份", "alias": []}], "class": "组织关系"}]}
    """
    D = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = json.loads(line.strip())
            target_list = []
            entity_dict = {}
            for entity in line["entity_list"]:
                if entity['type'] not in entity_dict:
                    entity_dict[entity['type']] = []
                entity_dict[entity['type']].append(entity)
                
            for entity_type in entity_dict:
                tmp_dict = {
                    'type':entity_type,
                    'role_list':[]
                }
                for entity in entity_dict[entity_type]:
                    tmp_dict['role_list'].append({
                        'role':entity_type,
                        'argument':entity['entity'],
                        'type':'',
                        'argument_start_index':entity['start_idx']
                    })
                target_list.append(tmp_dict)
            
#             for entity_type in entity_dict:
#                 for entity in entity_dict[entity_type]:
#                     tmp_dict = {
#                         'type':entity_type,
#                         'role_list':[]
#                     }
                
#                     tmp_dict['role_list'].append({
#                         'role':entity_type,
#                         'argument':entity['entity'],
#                         'type':'',
#                         'argument_start_index':entity['start_idx']
#                     })
#                     target_list.append(tmp_dict)
               
            D.append({
                "text":line["text"],
                "target_list":target_list
            })
        random.shuffle(D)
        print(filename, '=====size of data=====', len(D))
        return D
            
def load_duie(filename):
    """
    from {"text": "产后抑郁症@区分产后抑郁症与轻度情绪失调（产后忧郁或“婴儿忧郁”）是重要的，因为轻度情绪失调不需要治疗。", "spo_list": [{"Combined": false, "predicate": "鉴别诊断", "subject": "产后抑郁症", "subject_type": "疾病", "object": {"@value": "轻度情绪失调"}, "object_type": {"@value": "疾病"}}]}
    to 
    {"text": "消失的“外企光环”，5月份在华裁员900余人，香饽饽变“臭”了", "id": "cba11b5059495e635b4f95e7484b2684", "event_list": [{"event_type": "组织关系-裁员", "trigger": "裁员", "trigger_start_index": 15, "arguments": [{"argument_start_index": 17, "role": "裁员人数", "argument": "900余人", "alias": []}, {"argument_start_index": 10, "role": "时间", "argument": "5月份", "alias": []}], "class": "组织关系"}]}
    """
    D = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = json.loads(line.strip())
            target_list = []
            for spo in line["spo_list"]:
                event_dict = {
                    'type':spo['predicate'],
                    'role_list':[]
                }
                event_dict['role_list'].append({'role':spo['subject_type'], 'argument':spo['subject'], 'type':'subject-', 'argument_start_index':-1})
                for key in spo['object_type']:
                    event_dict['role_list'].append({'role':spo['object_type'][key], 'argument':spo['object'][key], 'argument_start_index':-1, 'type':'object-'})
                target_list.append(event_dict)
            D.append({
                "text":line["text"],
                "target_list":target_list
            })
        random.shuffle(D)
        print(filename, '=====size of data=====', len(D))
        return D
    
def load_duee(filename):
    """
    {"text": "消失的“外企光环”，5月份在华裁员900余人，香饽饽变“臭”了", "id": "cba11b5059495e635b4f95e7484b2684", "event_list": [{"event_type": "组织关系-裁员", "trigger": "裁员", "trigger_start_index": 15, "arguments": [{"argument_start_index": 17, "role": "裁员人数", "argument": "900余人", "alias": []}, {"argument_start_index": 10, "role": "时间", "argument": "5月份", "alias": []}], "class": "组织关系"}]}
    """
    D = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = json.loads(line.strip())
            d = {'text': line['text'], 'target_list': []}
            for e in line["event_list"]:
                event_dict = {
                    'type':e['event_type'],
                    'role_list':[]
                }
                if e.get('trigger', None):
                    event_dict['role_list'].append(
                        {'role':'触发词', 'argument':e['trigger'], 'type':'', 'argument_start_index':e.get('trigger_start_index', -1)}
                    )
                for a in e['arguments']:
                     event_dict['role_list'].append((
                         {'role':a['role'], 'argument':a['argument'], 'type':'', 'argument_start_index':a.get('argument_start_index', -1)}
                    ))
                d['target_list'].append(event_dict)
            D.append(d)
        random.shuffle(D)
        print(filename, '=====size of data=====', len(D))
        return D
    
import re
    
def load_entity_split(filename):
    """
    {"text": "对儿童SARST细胞亚群的研究表明，与成人SARS相比，儿童细胞下降不明显，证明上述推测成立。", "entity_list": [{"start_idx": 3, "end_idx": 9, "type": "身体部位", "entity": "SARST细胞"}, {"start_idx": 19, "end_idx": 24, "type": "疾病", "entity": "成人SARS"}]}
    to
     {"text": "消失的“外企光环”，5月份在华裁员900余人，香饽饽变“臭”了", "id": "cba11b5059495e635b4f95e7484b2684", "event_list": [{"event_type": "组织关系-裁员", "trigger": "裁员", "trigger_start_index": 15, "arguments": [{"argument_start_index": 17, "role": "裁员人数", "argument": "900余人", "alias": []}, {"argument_start_index": 10, "role": "时间", "argument": "5月份", "alias": []}], "class": "组织关系"}]}
    """
    D = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = json.loads(line.strip())
            target_list = []
            entity_dict = {}
            for entity in line["entity_list"]:
                if entity['type'] not in entity_dict:
                    entity_dict[entity['type']] = []
                entity_dict[entity['type']].append(entity)
                
            # for entity_type in entity_dict:
            #     tmp_dict = {
            #         'type':entity_type,
            #         'role_list':[]
            #     }
            #     for entity in entity_dict[entity_type]:
            #         tmp_dict['role_list'].append({
            #             'role':entity_type,
            #             'argument':entity['entity'],
            #             'type':'',
            #             'argument_start_index':entity['start_idx']
            #         })
            #     target_list.append(tmp_dict)
            
            for entity_type in entity_dict:
                for entity in entity_dict[entity_type]:
                    tmp_dict = {
                        'type':entity_type,
                        'role_list':[]
                    }
                
                    tmp_dict['role_list'].append({
                        'role':entity_type,
                        'argument':entity['entity'],
                        'type':'',
                        'argument_start_index':entity['start_idx']
                    })
                    target_list.append(tmp_dict)
               
            D.append({
                "text":line["text"],
                "target_list":target_list
            })
            
            for text in [line['text']] + re.split('[\n。]', line['text']):
                D.append({
                    "text":text,
                    "target_list":target_list
                })
        random.shuffle(D)
        print(filename, '=====size of data=====', len(D))
        return D
    
def load_duie_pos(filename):
    """
    {"text": "消失的“外企光环”，5月份在华裁员900余人，香饽饽变“臭”了", "id": "cba11b5059495e635b4f95e7484b2684", "event_list": [{"event_type": "组织关系-裁员", "trigger": "裁员", "trigger_start_index": 15, "arguments": [{"argument_start_index": 17, "role": "裁员人数", "argument": "900余人", "alias": []}, {"argument_start_index": 10, "role": "时间", "argument": "5月份", "alias": []}], "class": "组织关系"}]}
    """
    D = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = json.loads(line.strip())
            d = {'text': line['text'], 'target_list': []}
            for spo in line["spo_list"]:
                spo_dict = {
                    'type':spo['predicate'],
                    'role_list':[]
                }
                for a in spo['arguments']:
                     spo_dict['role_list'].append((
                         {'role':a['role'], 'argument':a['argument'], 'type':a.get('type', ''), 'argument_start_index':a.get('argument_start_index', -1)}
                    ))
                d['target_list'].append(spo_dict)
            D.append(d)
        random.shuffle(D)
        print(filename, '=====size of data=====', len(D), D[0])
        return D
            
def load_duie_split(filename):
    """
    from {"text": "产后抑郁症@区分产后抑郁症与轻度情绪失调（产后忧郁或“婴儿忧郁”）是重要的，因为轻度情绪失调不需要治疗。", "spo_list": [{"Combined": false, "predicate": "鉴别诊断", "subject": "产后抑郁症", "subject_type": "疾病", "object": {"@value": "轻度情绪失调"}, "object_type": {"@value": "疾病"}}]}
    to 
    {"text": "消失的“外企光环”，5月份在华裁员900余人，香饽饽变“臭”了", "id": "cba11b5059495e635b4f95e7484b2684", "event_list": [{"event_type": "组织关系-裁员", "trigger": "裁员", "trigger_start_index": 15, "arguments": [{"argument_start_index": 17, "role": "裁员人数", "argument": "900余人", "alias": []}, {"argument_start_index": 10, "role": "时间", "argument": "5月份", "alias": []}], "class": "组织关系"}]}
    """
    D = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = json.loads(line.strip())
            target_list = []
            for spo in line["spo_list"]:
                event_dict = {
                    'type':spo['predicate'],
                    'role_list':[]
                }
                event_dict['role_list'].append({'role':spo['subject_type'], 'argument':spo['subject'], 'type':'subject-', 'argument_start_index':-1})
                for key in spo['object_type']:
                    event_dict['role_list'].append({'role':spo['object_type'][key], 'argument':spo['object'][key], 'argument_start_index':-1, 'type':'object-'})
                target_list.append(event_dict)
            D.append({
                "text":line["text"],
                "target_list":target_list
            })
            for text in [line['text']] + re.split('[\n。]', line['text']):
                D.append({
                    "text":text,
                    "target_list":target_list
                })
        random.shuffle(D)
        print(filename, '=====size of data=====', len(D))
        return D
    
def load_duee_split(filename):
    """
    {"text": "消失的“外企光环”，5月份在华裁员900余人，香饽饽变“臭”了", "id": "cba11b5059495e635b4f95e7484b2684", "event_list": [{"event_type": "组织关系-裁员", "trigger": "裁员", "trigger_start_index": 15, "arguments": [{"argument_start_index": 17, "role": "裁员人数", "argument": "900余人", "alias": []}, {"argument_start_index": 10, "role": "时间", "argument": "5月份", "alias": []}], "class": "组织关系"}]}
    """
    D = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = json.loads(line.strip())
            d = {'text': line['text'], 'target_list': []}
            for e in line["event_list"]:
                event_dict = {
                    'type':e['event_type'],
                    'role_list':[]
                }
                if e.get('trigger', None):
                    event_dict['role_list'].append(
                        {'role':'触发词', 'argument':e['trigger'], 'type':'', 'argument_start_index':e.get('trigger_start_index', -1)}
                    )
                for a in e['arguments']:
                     event_dict['role_list'].append((
                         {'role':a['role'], 'argument':a['argument'], 'type':'', 'argument_start_index':a.get('argument_start_index', -1)}
                    ))
                d['target_list'].append(event_dict)
            D.append(d)
            for text in [line['text']] + re.split('[\n。]', line['text']):
                D.append({
                    "text":text,
                    "target_list":d['target_list']
                })
        random.shuffle(D)
        print(filename, '=====size of data=====', len(D))
        return D
    
def load_squad_style_data(filename):
    """See base class."""
    D = []
    data = json.load(open(filename, 'r', encoding='utf-8'))
    for content in data['data']:
        for para in content['paragraphs']:
            context = para['context']
            qas = para['qas']
            for qa in qas:
                question = qa['question']
                
                d = {
                    'text_a': context,
                    'text_b': question,
                    'target_list':[]
                }
                
                for anss in qa['answers']:
                    if isinstance(anss, list):
                        for ans in anss:
                            tmp_dict = {
                                'type':'答案',
                                'role_list': []
                            }
                            ans_text = ans.get('text', '')
                            tmp_dict['role_list'].append({
                                'type':'',
                                'argument':ans_text,
                                'start_index':ans.get('answer_start', -1),
                                'role':'答案'
                            })
                            d['target_list'].append(tmp_dict)
                    else:
                        ans_text = anss.get('text', '')
                        tmp_dict = {
                                'type':'答案',
                                'role_list': []
                            }
                        tmp_dict['role_list'].append({
                                    'type':'',
                                    'argument':ans_text,
                                    'start_index':anss.get('answer_start', -1),
                                    'role':'答案'
                        })
                        d['target_list'].append(tmp_dict)
                    
            D.append(d)
    return D
    
from functools import reduce
def deleteDuplicate_v1(input_dict_lst):
    f = lambda x,y:x if y in x else x + [y]
    return reduce(f, [[], ] + input_dict_lst)

def char_span_to_token_span(char2token, char_span):
    token_indexes = char2token[char_span[0]:char_span[1]]
    token_indexes = list(filter(partial(is_not, None), token_indexes))
    if token_indexes:
        return token_indexes[0], token_indexes[-1] + 1  # [start, end)
    else:  # empty
        return 0, 0

def token_span_to_char_span(token2char, token_span):
    char_indexes = token2char[token_span[0]:token_span[1]]
    char_indexes = [span for span in char_indexes]  # 删除CLS/SEP对应的span
    start, end = char_indexes[0][0], char_indexes[-1][1]
    return start, end

def get_token2char_char2token(tokenizer, text, maxlen):
    tokend = tokenizer(text, return_offsets_mapping=True, max_length=maxlen, truncation=True)
    token2char = tokend.offset_mapping
    
    char2token = [None] * len(text)
    for i, ((start, end)) in enumerate(token2char):
        char2token[start:end] = [i] * (end - start)
    
    return token2char, char2token

def sequence_padding(inputs, length=None, value=0, seq_dims=1, mode='post'):
    """Numpy函数，将序列padding到同一长度
    """
    if length is None:
        length = np.max([np.shape(x)[:seq_dims] for x in inputs], axis=0)
    elif not hasattr(length, '__getitem__'):
        length = [length]

    slices = [np.s_[:length[i]] for i in range(seq_dims)]
    slices = tuple(slices) if len(slices) > 1 else slices[0]
    pad_width = [(0, 0) for _ in np.shape(inputs[0])]

    outputs = []
    for x in inputs:
        x = x[slices]
        for i in range(seq_dims):
            if mode == 'post':
                pad_width[i] = (0, length[i] - np.shape(x)[i])
            elif mode == 'pre':
                pad_width[i] = (length[i] - np.shape(x)[i], 0)
            else:
                raise ValueError('"mode" argument must be "post" or "pre".')
        x = np.pad(x, pad_width, 'constant', constant_values=value)
        outputs.append(x)
    return np.array(outputs)

def search(pattern, sequence):
    """从sequence中寻找子串pattern
    如果找到，返回第一个下标；否则返回-1。
    """
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            return i
    return -1

def search_all(pattern, sequence):
    all_index = []
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            all_index.append(i)
    return all_index

def search_bin(bins, size):
    idx = len(bins) - 1
    for i, bin in enumerate(bins):
        if size <= bin:
            idx = i
            break
    return idx

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


