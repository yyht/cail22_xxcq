# -*- coding: utf-8 -*-
"""
@Auth: Xhw
@Description: CHIP/CBLUE 医学实体关系抽取，数据来源 https://tianchi.aliyun.com/dataset/dataDetail?dataId=95414
"""
import json
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import unicodedata
from copy import deepcopy

def load_name(filename):
    #{"text": "产后抑郁症@区分产后抑郁症与轻度情绪失调（产后忧郁或“婴儿忧郁”）是重要的，因为轻度情绪失调不需要治疗。", "spo_list": [{"Combined": false, "predicate": "鉴别诊断", "subject": "产后抑郁症", "subject_type": "疾病", "object": {"@value": "轻度情绪失调"}, "object_type": {"@value": "疾病"}}]}
    D = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = json.loads(line)
            D.append({
                "text":line["text"],
                "spo_list":line["spo_list"]
            })
        return D

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

def _is_control(ch):
    """控制类字符判断
    """
    return unicodedata.category(ch) in ('Cc', 'Cf')

def _is_special(ch):
    """判断是不是有特殊含义的符号
    """
    return bool(ch) and (ch[0] == '[') and (ch[-1] == ']')

def recover_bert_token(token):
    """获取token的“词干”（如果是##开头，则自动去掉##）
    """
    if token[:2] == '##':
        return token[2:]
    else:
        return token

def get_token_mapping(text, tokens, additional_special_tokens=set(), is_mapping_index=True):
    """给出原始的text和tokenize后的tokens的映射关系"""
    raw_text = deepcopy(text)
    text = text.lower()

    normalized_text, char_mapping = '', []
    for i, ch in enumerate(text):
        ch = unicodedata.normalize('NFD', ch)
        ch = ''.join([c for c in ch if unicodedata.category(c) != 'Mn'])
        ch = ''.join([
            c for c in ch
            if not (ord(c) == 0 or ord(c) == 0xfffd or _is_control(c))
        ])
        normalized_text += ch
        char_mapping.extend([i] * len(ch))

    text, token_mapping, offset = normalized_text, [], 0
    for token in tokens:
        token = token.lower()
        if token == '[unk]' or token in additional_special_tokens:
            if is_mapping_index:
                token_mapping.append(char_mapping[offset:offset+1])
            else:
                token_mapping.append(raw_text[offset:offset+1])
            offset = offset + 1
        elif _is_special(token):
            # 如果是[CLS]或者是[SEP]之类的词，则没有对应的映射
            token_mapping.append([])
        else:
            token = recover_bert_token(token)
            start = text[offset:].index(token) + offset
            end = start + len(token)
            if is_mapping_index:
                token_mapping.append(char_mapping[start:end])
            else:
                token_mapping.append(raw_text[start:end])
            offset = end

    return token_mapping

class data_generator(Dataset):
    def __init__(self, data, tokenizer, max_len, schema):
        self.total_data = data
        self.data, self.dev_data = train_test_split(self.total_data, test_size=0.1, random_state=42)
        
        print(len(self.data), "====size of train data====")
        
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.schema = schema #spo
    def __len__(self):
        return len(self.data)
    
    def encoder(self, item):
        text = item['text']
        tokens = self.tokenizer.tokenize(item['text'])
        token_mapping = get_token_mapping(item['text'], tokens)
        
        start_mapping = {j[0]: i for i, j in enumerate(token_mapping) if j}
        end_mapping = {j[-1]: i for i, j in enumerate(token_mapping) if j}
        
        spoes = set()
        entity_labels = []
        head_labels = []
        tail_labels = []
        for spo_dict in item['spo_list']:
            add_flag = False
            s = spo_dict['subject']
            o = spo_dict['object']
            s_t = spo_dict['subject_type']
            o_t = spo_dict['object_type']

            schema_key = spo_dict["subject_type"]+"_"+spo_dict["predicate"]+"_"+spo_dict["object_type"]['@value']
            p = self.schema[schema_key]

            s_pos = spo_dict['subject_pos']
            o_pos = spo_dict['object_pos']
                
            if s_pos[0] in start_mapping and s_pos[1] in end_mapping:
                s_start = start_mapping[s_pos[0]] + 1 # add pos of cls
                s_end = end_mapping[s_pos[1]] + 1 # add pos of cls
            else:
                continue

            if o_pos[0] in start_mapping and o_pos[1] in end_mapping:
                o_start = start_mapping[o_pos[0]] + 1 # add pos of cls
                o_end = end_mapping[o_pos[1]] + 1 # add pos of cls
            else:
                continue
                
            spoes.add((s_start, s_end, p, o_start, o_end))

        entity_labels = [set() for i in range(2)]
        head_labels = [set() for i in range(len(self.schema))]
        tail_labels = [set() for i in range(len(self.schema))]
        for sh, st, p, oh, ot in spoes:
            entity_labels[0].add((sh, st)) #实体提取：2个类型，头实体or尾实体
            entity_labels[1].add((oh, ot))
            head_labels[p].add((sh, oh)) #类似TP-Linker
            tail_labels[p].add((st, ot))
            add_flag = True
        for label in entity_labels+head_labels+tail_labels:
            if not label:
                label.add((0,0))

        entity_empty = False
        head_empty = False
        tail_empty = False
        for label in entity_labels:
            if not label:
                entity_empty = True
                break
        for label in head_labels:
            if not label:
                head_empty = True
                break
        for label in tail_labels:
            if not label:
                tail_empty = True
                break

        valid_flag = True
        if add_flag and not entity_empty and not head_empty and not tail_empty:
            entity_labels = sequence_padding([list(l) for l in entity_labels])
            head_labels = sequence_padding([list(l) for l in head_labels])
            tail_labels = sequence_padding([list(l) for l in tail_labels])
        else:
            entity_labels, head_labels, tail_labels = [], [], []
            valid_flag = False
            
        tokens = ['[CLS]'] + tokens + ["[SEP]"]
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            
        attention_mask = [1]*len(input_ids)
        token_type_ids = [0]*len(input_ids)
        
        return [valid_flag, text, 
                entity_labels, head_labels, tail_labels, 
                input_ids, attention_mask, token_type_ids]

    def __getitem__(self, idx):
        item = self.data[idx]
        return self.encoder(item)

    @staticmethod
    def collate(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids = [], [], []
        batch_entity_labels, batch_head_labels, batch_tail_labels = [], [], []
        text_list = []
        for item in examples:
            [valid_flag, text, entity_labels, 
             head_labels, tail_labels, input_ids, attention_mask, token_type_ids] = item
            if not valid_flag:
                continue
            batch_entity_labels.append(entity_labels)
            batch_head_labels.append(head_labels)
            batch_tail_labels.append(tail_labels)
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            text_list.append(text)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_entity_labels = torch.tensor(sequence_padding(batch_entity_labels, seq_dims=2)).long()
        batch_head_labels = torch.tensor(sequence_padding(batch_head_labels, seq_dims=2)).long()
        batch_tail_labels = torch.tensor(sequence_padding(batch_tail_labels, seq_dims=2)).long()\
        
        return text_list, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels, batch_head_labels, batch_tail_labels


