# -*- coding: utf-8 -*-
"""
@Auth: Xhw
@Description: CHIP/CBLUE 医学实体关系抽取，数据来源 https://tianchi.aliyun.com/dataset/dataDetail?dataId=95414
"""
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import random
import itertools

def load_name(filename):
    #{"text": "产后抑郁症@区分产后抑郁症与轻度情绪失调（产后忧郁或“婴儿忧郁”）是重要的，因为轻度情绪失调不需要治疗。", "spo_list": [{"Combined": false, "predicate": "鉴别诊断", "subject": "产后抑郁症", "subject_type": "疾病", "object": {"@value": "轻度情绪失调"}, "object_type": {"@value": "疾病"}}]}
    D = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = json.loads(line)
            spo_list = []
            for spo in line["spo_list"]:
                for key in spo['object_type']:
                    spo_list.append((spo["subject"], spo["predicate"], spo["object"][key], spo["subject_type"], spo["object_type"][key]))
            D.append({
                "text":line["text"],
                "spo_list":spo_list
            })
            # D.append({
            #     "text":line["text"],
            #     "spo_list":[(spo["subject"], spo["predicate"], spo["object"]["@value"], spo["subject_type"], spo["object_type"]["@value"])
            #                 for spo in line["spo_list"]]
            # })
        # random.shuffle(D)
        print(D[0:1], '==filename==', filename)
        return D

import re
def load_name_split(filename):
    #{"text": "产后抑郁症@区分产后抑郁症与轻度情绪失调（产后忧郁或“婴儿忧郁”）是重要的，因为轻度情绪失调不需要治疗。", "spo_list": [{"Combined": false, "predicate": "鉴别诊断", "subject": "产后抑郁症", "subject_type": "疾病", "object": {"@value": "轻度情绪失调"}, "object_type": {"@value": "疾病"}}]}
    D = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = json.loads(line)
            spo_list = []
            for spo in line["spo_list"]:
                for key in spo['object_type']:
                    spo_list.append((spo["subject"], spo["predicate"], spo["object"][key], spo["subject_type"], spo["object_type"][key]))
            D.append({
                "text":line["text"],
                "spo_list":spo_list
            })
            for text in re.split('\n', line["text"]):
                if len(text) >= 2:
                    D.append({
                        "text":text,
                        "spo_list":spo_list
                    })
            # D.append({
            #     "text":line["text"],
            #     "spo_list":[(spo["subject"], spo["predicate"], spo["object"]["@value"], spo["subject_type"], spo["object_type"]["@value"])
            #                 for spo in line["spo_list"]]
            # })
        # random.shuffle(D)
        print(D[0:1], '==filename==', filename)
        return D

def load_name_pos(filename):
    #{"text": "产后抑郁症@区分产后抑郁症与轻度情绪失调（产后忧郁或“婴儿忧郁”）是重要的，因为轻度情绪失调不需要治疗。", "spo_list": [{"Combined": false, "predicate": "鉴别诊断", "subject": "产后抑郁症", "subject_type": "疾病", "object": {"@value": "轻度情绪失调"}, "object_type": {"@value": "疾病"}}]}
    D = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = json.loads(line)
            spo_list = []
            for spo in line["spo_list"]:
                for key in spo['object_type']:
                    # if not spo["subject"] or not spo["object"][key]:
                    #     print(line, '=================')
                    #     continue
                    spo_list.append((spo["subject"], spo["predicate"], spo["object"][key], spo["subject_type"], spo["object_type"][key]))
            # if not spo_list:
            #     print(line, '=================')
            #     continue
            D.append({
                "text":line["text"],
                "spo_list":spo_list
            })
            # D.append({
            #     "text":line["text"],
            #     "spo_list":[(spo["subject"], spo["predicate"], spo["object"]["@value"], spo["subject_type"], spo["object_type"]["@value"])
            #                 for spo in line["spo_list"]]
            # })
        # random.shuffle(D)
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

def flat_list(h_list):
    e_list = []

    for item in h_list:
        if isinstance(item, list):
            e_list.extend(flat_list(item))
        else:
            e_list.append(item)
    return e_list

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

import re
class data_generator(Dataset):
    def __init__(self, data, tokenizer, max_len, schema, label=0, seg_token='<S>'):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.schema = schema #spo
        self.labels = [label] * len(self.data)
        self.seg_token = seg_token
        
        self.tokenizer.add_special_tokens({
                    "additional_special_tokens": [self.seg_token] })
        
    def __len__(self):
        return len(self.data)

    def encoder(self, item):
        text = item["text"]
        
        encoder_text = self.tokenizer(text, return_offsets_mapping=True, max_length=self.max_len, truncation=True)
        input_ids = encoder_text["input_ids"]
        token_type_ids = encoder_text["token_type_ids"] #RoBERTa不需要NSP任务
        attention_mask = encoder_text["attention_mask"]
        spoes = set()
        for s, p, o, s_t, o_t in item["spo_list"]:
            if s and o:
                s = self.tokenizer.encode(s, add_special_tokens=False)
                p = self.schema[s_t + "_" + p + "_" +o_t]
                o = self.tokenizer.encode(o, add_special_tokens=False)
                # sh = search(s, input_ids)
                # oh = search(o, input_ids)
                # if sh != -1 and oh != -1:
                #     spoes.add((sh, sh+len(s)-1, p, oh, oh+len(o)-1))
                sh_list = search_all(s, input_ids)
                oh_list = search_all(o, input_ids)
                if sh_list and oh_list:
                    for sh in sh_list:
                        for oh in oh_list:
                            spoes.add((sh, sh+len(s)-1, p, oh, oh+len(o)-1))
        entity_labels = [set() for i in range(2)]
        head_labels = [set() for i in range(len(self.schema))]
        tail_labels = [set() for i in range(len(self.schema))]
        
        for sh, st, p, oh, ot in spoes:
            entity_labels[0].add((sh, st)) #实体提取：2个类型，头实体or尾实体
            entity_labels[1].add((oh, ot))
            head_labels[p].add((sh, oh)) #类似TP-Linker
            tail_labels[p].add((st, ot))
        for label in entity_labels+head_labels+tail_labels:
            if not label:
                label.add((0,0))
        # 例如entity = [{(1,3)}, {(4,5), (7,9)}]
        # entity[0]即{(1,3)}代表头实体首尾， entity[1]即{(4,5),{7,9}}代表尾实体首尾
        # 需要标签对齐为 [[[1,3][0,0]] , [[4,5][7,9]]]
        entity_labels = sequence_padding([list(l) for l in entity_labels])
        head_labels = sequence_padding([list(l) for l in head_labels])
        tail_labels = sequence_padding([list(l) for l in tail_labels])

        return text, entity_labels, head_labels, tail_labels, input_ids, attention_mask, token_type_ids

    def __getitem__(self, idx):
        item = self.data[idx]
        return self.encoder(item)
    
    def get_labels(self):
        return self.labels

    @staticmethod
    def collate(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids = [], [], []
        batch_entity_labels, batch_head_labels, batch_tail_labels = [], [], []
        text_list = []
        for item in examples:
            text, entity_labels, head_labels, tail_labels, input_ids, attention_mask, token_type_ids = item
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

import re
class data_generator_nce(Dataset):
    def __init__(self, data, tokenizer, max_len, schema, predicate_schema, label=0, seg_token='<S>'):
        self.data = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.schema = schema #spo
        self.predicate_schema = predicate_schema
        
        neg_cnt = 0
        
        for item in data:
            spo_dict = {}
            for s, p, o, s_t, o_t in item["spo_list"]:
                predicate = s_t + "_" + p + "_" +o_t
                if predicate not in spo_dict:
                    spo_dict[predicate] = []
                spo_dict[predicate].append((s, p, o, s_t, o_t))
                                
            for predicate in spo_dict:
                tmp_dict = {
                    'text':item['text'],
                    'spo_list': spo_dict[predicate],
                    'predicate': predicate
                }
                self.data.append(tmp_dict)
                
            neg_e = list(set(list(self.predicate_schema.keys())) - set(list(spo_dict.keys())))
            import random
            random.shuffle(neg_e)
            
            if len(neg_e) >= 1:
                tmp_dict = {
                        'text':item['text'],
                        'spo_list':[],
                        'candidate_type': neg_e

                }
                self.data.append(tmp_dict)
                neg_cnt += 1
                
        print('==total negative counter==', neg_cnt)
        
        self.labels = [label] * len(self.data)
        self.seg_token = seg_token
        
        self.tokenizer.add_special_tokens({
                    "additional_special_tokens": [self.seg_token] })
        
    def __len__(self):
        return len(self.data)

    def encoder(self, item):
        text = item["text"]
        
        if 'candidate_type' in item:
            import random
            random.shuffle(item['candidate_type'])
            predicate = item['candidate_type'][0]
        else:
            predicate = item['predicate']
        
        encoder_text = self.tokenizer(text, return_offsets_mapping=True, max_length=self.max_len, truncation=True)
        input_ids = encoder_text["input_ids"]
        token_type_ids = encoder_text["token_type_ids"] #RoBERTa不需要NSP任务
        attention_mask = encoder_text["attention_mask"]
        spoes = set()
        for s, p, o, s_t, o_t in item["spo_list"]:
            if s and o:
                s = self.tokenizer.encode(s, add_special_tokens=False)
                p = self.schema[s_t + "_" + p + "_" +o_t]
                o = self.tokenizer.encode(o, add_special_tokens=False)
                # sh = search(s, input_ids)
                # oh = search(o, input_ids)
                # if sh != -1 and oh != -1:
                #     spoes.add((sh, sh+len(s)-1, p, oh, oh+len(o)-1))
                sh_list = search_all(s, input_ids)
                oh_list = search_all(o, input_ids)
                if sh_list and oh_list:
                    for sh in sh_list:
                        for oh in oh_list:
                            spoes.add((sh, sh+len(s)-1, p, oh, oh+len(o)-1))
                            
        entity_labels = [set() for i in range(2)]
        head_labels = [set() for i in range(1)]
        tail_labels = [set() for i in range(1)]
        
        for sh, st, p, oh, ot in spoes:
            entity_labels[0].add((sh, st)) #实体提取：2个类型，头实体or尾实体
            entity_labels[1].add((oh, ot))
            head_labels[0].add((sh, oh)) #类似TP-Linker
            tail_labels[0].add((st, ot))
        for label in entity_labels+head_labels+tail_labels:
            if not label:
                label.add((0,0))
        # 例如entity = [{(1,3)}, {(4,5), (7,9)}]
        # entity[0]即{(1,3)}代表头实体首尾， entity[1]即{(4,5),{7,9}}代表尾实体首尾
        # 需要标签对齐为 [[[1,3][0,0]] , [[4,5][7,9]]]
        entity_labels = sequence_padding([list(l) for l in entity_labels])
        head_labels = sequence_padding([list(l) for l in head_labels])
        tail_labels = sequence_padding([list(l) for l in tail_labels])
        
        et_text = self.tokenizer('{}'.format(predicate), return_offsets_mapping=True, max_length=64)
        input_ids += et_text['input_ids'][1:]
        token_type_ids +=  et_text['token_type_ids'][1:]
        attention_mask += et_text['attention_mask'][1:]

        return text, entity_labels, head_labels, tail_labels, input_ids, attention_mask, token_type_ids

    def __getitem__(self, idx):
        item = self.data[idx]
        return self.encoder(item)
    
    def get_labels(self):
        return self.labels

    @staticmethod
    def collate(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids = [], [], []
        batch_entity_labels, batch_head_labels, batch_tail_labels = [], [], []
        text_list = []
        for item in examples:
            text, entity_labels, head_labels, tail_labels, input_ids, attention_mask, token_type_ids = item
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
    
    
class data_generator_qa_subject(Dataset):
    def __init__(self, data, tokenizer, max_len, schema, task_dict, label=0):
        
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.schema = schema #spo
        self.task_dict = task_dict
        
        self.sentinel_token = self.task_dict['sentinel_token']
        self.sep_token = self.task_dict['sep_token']
        
        self.tokenizer.add_special_tokens({
                    "additional_special_tokens": [ self.sentinel_token.format(i) for i in range(1, 100)]+[self.sep_token] })
        
        self.data = []
        for item in data:
            # (spo["subject"], spo["predicate"], spo["object"][key], spo["subject_type"], spo["object_type"][key])
            spo_dict = {}
            for spo in item['spo_list']:
                predicate = spo[3]+'_'+spo[1]+'_'+spo[4]
                if predicate not in spo_dict:
                    spo_dict[predicate] = set()
                spo_dict[predicate].add(spo) # add subject and predicate
            for predicate in spo_dict:
                tmp_item = {
                    'text': item['text'],
                    'spo_list': list(spo_dict[predicate])
                }
                self.data.append(tmp_item)
        self.labels = [label] * len(self.data)
        
    def __len__(self):
        return len(self.data)

    def encoder(self, item):
        text = item["text"]
        
        encoder_text = self.tokenizer(text, return_offsets_mapping=True, max_length=self.max_len, truncation=True)
        input_ids = encoder_text["input_ids"]
        token_type_ids = encoder_text["token_type_ids"] #RoBERTa不需要NSP任务
        attention_mask = encoder_text["attention_mask"]
        spoes = set()
        for s, p_, o, s_t, o_t in item["spo_list"]:
            p =  s_t+'_'+p_+'_'+o_t
            if s:
                s = self.tokenizer.encode(s, add_special_tokens=False)
                sh_list = search_all(s, input_ids)
                if sh_list:
                    for sh in sh_list:
                        spoes.add((sh, sh+len(s)-1, p))
        
        entity_labels = [set() for i in range(1)]
        
        for sh, st, p in spoes:
            entity_labels[0].add((sh, st)) #实体提取：2个类型，头实体or尾实体
        for label in entity_labels:
            if not label:
                label.add((0,0))
        # 例如entity = [{(1,3)}, {(4,5), (7,9)}]
        # entity[0]即{(1,3)}代表头实体首尾， entity[1]即{(4,5),{7,9}}代表尾实体首尾
        # 需要标签对齐为 [[[1,3][0,0]] , [[4,5][7,9]]]
        entity_labels = sequence_padding([list(l) for l in entity_labels])
        
        [subject_type, predicate_type, object_type] = p.split('_')
        
        prefix = self.task_dict['prefix']
        suffix = self.task_dict['suffix']
        
        # et_text = self.tokenizer('{}关系类型{}{}'.format('', subject_type+prefix, predicate_type+suffix+object_type),
                                 # return_offsets_mapping=True, max_length=self.max_len)
        et_text = self.tokenizer('{}{}{}'.format(prefix, predicate_type, suffix),
                                 return_offsets_mapping=True, max_length=self.max_len)
        input_ids += et_text['input_ids'][1:]
        token_type_ids +=  et_text['token_type_ids'][1:]
        attention_mask += et_text['attention_mask'][1:]
        
        at_text = self.tokenizer('{}'.format(subject_type), return_offsets_mapping=True, max_length=self.max_len)
        input_ids += at_text['input_ids'][1:]
        token_type_ids +=  at_text['token_type_ids'][1:]
        attention_mask += at_text['attention_mask'][1:]

        return text, entity_labels, input_ids, attention_mask, token_type_ids

    def __getitem__(self, idx):
        item = self.data[idx]
        return self.encoder(item)
    
    def get_labels(self):
        return self.labels

    @staticmethod
    def collate(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids = [], [], []
        batch_entity_labels = []
        text_list = []
        for item in examples:
            text, entity_labels, input_ids, attention_mask, token_type_ids = item
            batch_entity_labels.append(entity_labels)
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            text_list.append(text)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_entity_labels = torch.tensor(sequence_padding(batch_entity_labels, seq_dims=2)).long()
        
        return text_list, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels
    
class data_generator_qa_sp_object(Dataset):
    def __init__(self, data, tokenizer, max_len, schema, task_dict, label=0):
        
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.schema = schema #spo
        
        self.task_dict = task_dict
        
        self.sentinel_token = self.task_dict['sentinel_token']
        self.sep_token = self.task_dict['sep_token']
        
        self.tokenizer.add_special_tokens({
                    "additional_special_tokens": [ self.sentinel_token.format(i) for i in range(1, 100)]+[self.sep_token] })
        
        self.data = []
        for item in data:
            # (spo["subject"], spo["predicate"], spo["object"][key], spo["subject_type"], spo["object_type"][key])
            spo_dict = {}
            for spo in item['spo_list']:
                predicate = spo[3]+'_'+spo[1]+'_'+spo[4]
                if (spo[0], predicate) not in spo_dict:
                    spo_dict[(spo[0], predicate)] = set()
                spo_dict[(spo[0], predicate)].add(spo) # add subject and predicate
            for (subject, predicate) in spo_dict:
                tmp = {
                    'text': item['text'],
                    'spo_list': list(spo_dict[(subject, predicate)])
                }
                self.data.append(tmp)
                for p in self.schema:
                    if p == predicate:
                        continue
                    subject_type, predicate_type, object_type = p.split('_')
                    tmp = {
                        'text': item['text'],
                        'spo_list': [(subject, predicate_type, '', subject_type, object_type)]
                    }
                    self.data.append(tmp)
                
        self.labels = [label] * len(self.data)
        
    def __len__(self):
        return len(self.data)

    def encoder(self, item):
        text = item["text"]
        
        encoder_text = self.tokenizer(text, return_offsets_mapping=True, max_length=self.max_len, truncation=True)
        input_ids = encoder_text["input_ids"]
        token_type_ids = encoder_text["token_type_ids"] #RoBERTa不需要NSP任务
        attention_mask = encoder_text["attention_mask"]
        spoes = set()
        
        for s, p_, o, s_t, o_t in item["spo_list"]:
            p =  s_t+'_'+p_+'_'+o_t
            if s and o:
                s_ids = self.tokenizer.encode(s, add_special_tokens=False)
                o_ids = self.tokenizer.encode(o, add_special_tokens=False)
                sh_list = search_all(s_ids, input_ids)
                oh_list = search_all(o_ids, input_ids)
                if oh_list and sh_list:
                    for oh in oh_list:
                        spoes.add((oh, oh+len(o)-1, p))
                                
        entity_labels = [set() for i in range(1)]
        
        for oh, ot, p in spoes:
            entity_labels[0].add((oh, ot)) #实体提取：2个类型，头实体or尾实体
        for label in entity_labels:
            if not label:
                label.add((0,0))
        # 例如entity = [{(1,3)}, {(4,5), (7,9)}]
        # entity[0]即{(1,3)}代表头实体首尾， entity[1]即{(4,5),{7,9}}代表尾实体首尾
        # 需要标签对齐为 [[[1,3][0,0]] , [[4,5][7,9]]]
        entity_labels = sequence_padding([list(l) for l in entity_labels])
        
        [subject_type, predicate_type, object_type] = p.split('_')
        
        prefix = self.task_dict['prefix']
        suffix = self.task_dict['suffix']
        
        at_text = self.tokenizer('{}{}'.format('', s), return_offsets_mapping=True, max_length=self.max_len)
        input_ids += at_text['input_ids'][1:]
        token_type_ids +=  at_text['token_type_ids'][1:]
        attention_mask += at_text['attention_mask'][1:]
        
        et_text = self.tokenizer('{}{}{}'.format(prefix, predicate_type, suffix),
                                 return_offsets_mapping=True, max_length=self.max_len)
        input_ids += et_text['input_ids'][1:]
        token_type_ids +=  et_text['token_type_ids'][1:]
        attention_mask += et_text['attention_mask'][1:]
        
        ot_text = self.tokenizer('{}'.format(object_type), return_offsets_mapping=True, max_length=self.max_len)
        input_ids += ot_text['input_ids'][1:]
        token_type_ids +=  ot_text['token_type_ids'][1:]
        attention_mask += ot_text['attention_mask'][1:]

        return text, entity_labels, input_ids, attention_mask, token_type_ids

    def __getitem__(self, idx):
        item = self.data[idx]
        return self.encoder(item)
    
    def get_labels(self):
        return self.labels

    @staticmethod
    def collate(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids = [], [], []
        batch_entity_labels = []
        text_list = []
        for item in examples:
            text, entity_labels, input_ids, attention_mask, token_type_ids = item
            batch_entity_labels.append(entity_labels)
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            text_list.append(text)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_entity_labels = torch.tensor(sequence_padding(batch_entity_labels, seq_dims=2)).long()
        
        return text_list, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels
    
class data_generator_mention_detector(Dataset):
    def __init__(self, data, tokenizer, max_len, schema, label=0):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.schema = schema #spo
        self.labels = [label] * len(self.data)
        
    def __len__(self):
        return len(self.data)

    def encoder(self, item):
        text = item["text"]
        encoder_text = self.tokenizer(text, return_offsets_mapping=True, max_length=self.max_len, truncation=True)
        input_ids = encoder_text["input_ids"]
        token_type_ids = encoder_text["token_type_ids"] #RoBERTa不需要NSP任务
        attention_mask = encoder_text["attention_mask"]
        spoes = set()
        for s, p, o, s_t, o_t in item["spo_list"]:
            if s and o:
                s = self.tokenizer.encode(s, add_special_tokens=False)
                o = self.tokenizer.encode(o, add_special_tokens=False)
                # sh = search(s, input_ids)
                # oh = search(o, input_ids)
                # if sh != -1 and oh != -1:
                #     spoes.add((sh, sh+len(s)-1, p, oh, oh+len(o)-1))
                sh_list = search_all(s, input_ids)
                oh_list = search_all(o, input_ids)
                if sh_list and oh_list:
                    for sh in sh_list:
                        for oh in oh_list:
                            spoes.add((sh, sh+len(s)-1, s_t, o_t, oh, oh+len(o)-1))
        entity_labels = [set() for i in range(2)]
        head_labels = [set() for i in range(1)] # if (subj, obj) is a pair
        tail_labels = [set() for i in range(1)]
        
        for sh, st, s_t, o_t, oh, ot in spoes:
            entity_labels[0].add((sh, st)) #实体提取：2个类型，头实体or尾实体
            entity_labels[1].add((oh, ot))
            head_labels[0].add((sh, oh)) #类似TP-Linker
            tail_labels[0].add((st, ot))
        for label in entity_labels+head_labels+tail_labels:
            if not label:
                label.add((0,0))
        # 例如entity = [{(1,3)}, {(4,5), (7,9)}]
        # entity[0]即{(1,3)}代表头实体首尾， entity[1]即{(4,5),{7,9}}代表尾实体首尾
        # 需要标签对齐为 [[[1,3][0,0]] , [[4,5][7,9]]]
        entity_labels = sequence_padding([list(l) for l in entity_labels])
        head_labels = sequence_padding([list(l) for l in head_labels])
        tail_labels = sequence_padding([list(l) for l in tail_labels])

        return text, entity_labels, head_labels, tail_labels, input_ids, attention_mask, token_type_ids

    def __getitem__(self, idx):
        item = self.data[idx]
        return self.encoder(item)
    
    def get_labels(self):
        return self.labels

    @staticmethod
    def collate(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids = [], [], []
        batch_entity_labels, batch_head_labels, batch_tail_labels = [], [], []
        text_list = []
        for item in examples:
            text, entity_labels, head_labels, tail_labels, input_ids, attention_mask, token_type_ids = item
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
    
class data_generator_cls_then_extraction(Dataset):
    def __init__(self, data, tokenizer, max_len, schema,label=0,  mode='cls', seg_token='[SEP]', add_neg=False):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.schema = schema #spo
        
        self.mode = mode
        self.seg_token = seg_token
        self.add_neg = add_neg
        
        self.features = []
        for item in self.data:
            for feature in self.encoder(item):
                self.features.append(feature)

        self.labels = [label] * len(self.features)
        
    def __len__(self):
        return len(self.features)

    def encoder(self, item):
        text = item["text"]
        encoder_text = self.tokenizer(text, return_offsets_mapping=True, max_length=self.max_len, truncation=True)
        input_ids = encoder_text["input_ids"]
        token_type_ids = encoder_text["token_type_ids"] #RoBERTa不需要NSP任务
        attention_mask = encoder_text["attention_mask"]
        spoes = set()
        for s, p, o, s_t, o_t in item["spo_list"]:
            if s and o:
                s = self.tokenizer.encode(s, add_special_tokens=False)
                so_key = s_t + "_" + o_t
                p_key = s_t + "_" + p + "_" +o_t
                p_type = self.schema[s_t + "_" + p + "_" +o_t]
                o = self.tokenizer.encode(o, add_special_tokens=False)
                # sh = search(s, input_ids)
                # oh = search(o, input_ids)
                # if sh != -1 and oh != -1:
                #     spoes.add((sh, sh+len(s)-1, p, oh, oh+len(o)-1))
                sh_list = search_all(s, input_ids)
                oh_list = search_all(o, input_ids)
                if sh_list and oh_list:
                    for sh in sh_list:
                        for oh in oh_list:
                            spoes.add((sh, sh+len(s)-1, p_type, p_key, so_key, oh, oh+len(o)-1))
        
        output_tuple_list = []
        if self.mode == 'cls':
            labels = [0] * len(self.schema)
            for sh, st, p_type, p_key, so_key, oh, ot in spoes:
                labels[p_type] = 1
            output_tuple_list.append((text, labels, input_ids, attention_mask, token_type_ids))
        elif self.mode == 'nli':
            pos_p = set()
            for sh, st, p_type, p_key, so_key, oh, ot in spoes:
                p_ids = self.tokenizer(p_key.replace('_', self.seg_token))['input_ids'][1:] #  [cls] text [sep] relation [sep]
                new_input_ids = input_ids + p_ids    
                token_type_ids = [0]*len(new_input_ids)
                attention_mask = [1]*len(new_input_ids)
                output_tuple_list.append((text, 1, new_input_ids, attention_mask, token_type_ids))
                pos_p.add(p_key)
            
            left_key = list(self.schema - pos_p)
            random.shuffle(left_key)
            
            for p_key in left_key[:2]:
                p_ids = self.tokenizer(p_key.replace('_', self.seg_token))['input_ids'][1:] #  [cls] text [sep] relation [sep]
                new_input_ids = input_ids + p_ids    
                token_type_ids = [0]*len(new_input_ids)
                attention_mask = [1]*len(new_input_ids)
                output_tuple_list.append((text, 0, new_input_ids, attention_mask, token_type_ids))
            
        elif self.mode == 'contrast':
            pos_p = set()
            for sh, st, p_type, p_key, so_key, oh, ot in spoes:
                pos_p.add(p_key)
            
            left_key = list(self.schema - p_key)
            random.shuffle(left_key)
            for sh, st, p_type, p_key, so_key, oh, ot in spoes:
                neg_p = random.sample(left_key)
                
                encoder_pos_p = self.tokenizer(p_key.replace('_', self.seg_token)) #  [cls] text [sep] relation [sep]
                pos_input_ids = encoder_pos_p["input_ids"]
                pos_token_type_ids = encoder_pos_p["token_type_ids"] #RoBERTa不需要NSP任务
                pos_attention_mask = encoder_pos_p["attention_mask"]
                
                encoder_neg_p = self.tokenizer(neg_p.replace('_', self.seg_token)) #  [cls] text [sep] relation [sep]
                neg_input_ids = encoder_neg_p["input_ids"]
                neg_token_type_ids = encoder_neg_p["token_type_ids"] #RoBERTa不需要NSP任务
                neg_attention_mask = encoder_neg_p["attention_mask"]
            
                output_tuple_list.append(((text, p), 1, (input_ids, pos_input_ids, neg_input_ids), 
                                          (attention_mask, pos_attention_mask, neg_attention_mask), 
                                          (token_type_ids, pos_token_type_ids, neg_token_type_ids)))
        elif self.mode == 'p_extraction':
            spoes_p = {}
            for sh, st, p_type, p_key, so_key, oh, ot in spoes:
                if p_key not in spoes_p:
                    spoes_p[p_key] = set()
                spoes_p[p_key].add((sh, st, p_type, p_key, so_key, oh, ot))
                
            valid_key = set()
            
            for p_key in spoes_p:
                entity_labels = [set() for i in range(2)]
                head_labels = [set() for i in range(1)] # if (subj, obj) is a pair
                tail_labels = [set() for i in range(1)]
                valid_key.add(p_key)
                
                p_ids = self.tokenizer(p_key.replace('_', self.seg_token))['input_ids'][1:] #  [cls] text [sep] sub-type [sep] relation [sep] obj-type [sep]
                new_input_ids = input_ids + p_ids    
                token_type_ids = [0]*len(new_input_ids)
                attention_mask = [1]*len(new_input_ids)
                
                for sh, st, p_type, p_key, so_key, oh, ot in spoes_p[p_key]:
                    entity_labels[0].add((sh, st)) #实体提取：2个类型，头实体or尾实体
                    entity_labels[1].add((oh, ot))
                    head_labels[0].add((sh, oh)) #类似TP-Linker
                    tail_labels[0].add((st, ot))
                for label in entity_labels+head_labels+tail_labels:
                    if not label:
                        label.add((0,0))
                # 例如entity = [{(1,3)}, {(4,5), (7,9)}]
                # entity[0]即{(1,3)}代表头实体首尾， entity[1]即{(4,5),{7,9}}代表尾实体首尾
                # 需要标签对齐为 [[[1,3][0,0]] , [[4,5][7,9]]]
                entity_labels = sequence_padding([list(l) for l in entity_labels])
                head_labels = sequence_padding([list(l) for l in head_labels])
                tail_labels = sequence_padding([list(l) for l in tail_labels])
                output_tuple_list.append(((text, p_key), entity_labels, head_labels, tail_labels, new_input_ids, attention_mask, token_type_ids))
                
            if self.add_neg:
                left_key = set(list(self.schema.keys())) - valid_key
                left_key = list(left_key)
                random.shuffle(left_key)
                for p_key in left_key[:2]:

                    p_ids = self.tokenizer(p_key.replace('_', self.seg_token))['input_ids'][1:] #  [cls] text [sep] sub-type [sep] relation [sep] obj-type [sep]
                    new_input_ids = input_ids + p_ids    
                    token_type_ids = [0]*len(new_input_ids)
                    attention_mask = [1]*len(new_input_ids)

                    entity_labels = [set() for i in range(2)]
                    head_labels = [set() for i in range(1)] # if (subj, obj) is a pair
                    tail_labels = [set() for i in range(1)]
                    for label in entity_labels+head_labels+tail_labels:
                        if not label:
                            label.add((0,0))
                    entity_labels = sequence_padding([list(l) for l in entity_labels])
                    head_labels = sequence_padding([list(l) for l in head_labels])
                    tail_labels = sequence_padding([list(l) for l in tail_labels])
                    output_tuple_list.append(((text, p_key), entity_labels, head_labels, tail_labels, new_input_ids, attention_mask, token_type_ids))
                
        return output_tuple_list
       
    def __getitem__(self, idx):
        return self.features[idx]
    
    def get_labels(self):
        return self.labels

    @staticmethod
    def collate_cls_nli(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids = [], [], []
        
        batch_labels = []
        text_list = []
        for item in examples:
            text, labels, input_ids, attention_mask, token_type_ids = item
            batch_labels.append(labels)
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            text_list.append(text)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).long()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_labels = torch.tensor(batch_labels).long()
        
        return text_list, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_labels
    
    @staticmethod
    def collate_contrast(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids = [], [], []
        batch_pos_token_ids, batch_pos_mask_ids, batch_pos_token_type_ids = [], [], []
        batch_neg_token_ids, batch_neg_mask_ids, batch_neg_token_type_ids = [], [], []
        batch_labels = []
        text_list = []
        for item in examples:
            text, labels, input_ids, attention_mask, token_type_ids = item
            batch_labels.append(labels)
            batch_token_ids.append(input_ids[0])
            batch_mask_ids.append(attention_mask[0])
            batch_token_type_ids.append(token_type_ids[0])
            text_list.append(text)
            batch_pos_token_ids.append(input_ids[1])
            batch_pos_mask_ids.append(attention_mask[1])
            batch_pos_token_type_ids.append(token_type_ids[1])
            
            batch_neg_token_ids.append(input_ids[2])
            batch_neg_mask_ids.append(attention_mask[2])
            batch_neg_token_type_ids.append(token_type_ids[2])
            
        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        
        batch_pos_token_ids = torch.tensor(sequence_padding(batch_pos_token_ids)).long()
        batch_pos_mask_ids = torch.tensor(sequence_padding(batch_pos_mask_ids)).float()
        batch_pos_token_type_ids = torch.tensor(sequence_padding(batch_pos_token_type_ids)).long()#RoBERTa 不需要NSP
        
        batch_neg_token_ids = torch.tensor(sequence_padding(batch_neg_token_ids)).long()
        batch_neg_mask_ids = torch.tensor(sequence_padding(batch_neg_mask_ids)).float()
        batch_neg_token_type_ids = torch.tensor(sequence_padding(batch_neg_token_type_ids)).long()#RoBERTa 不需要NSP
        return text_list, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_labels
    
    @staticmethod
    def collate_p_extraction(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids = [], [], []
        batch_entity_labels, batch_head_labels, batch_tail_labels = [], [], []
        text_list = []
        for item in examples:
            text, entity_labels, head_labels, tail_labels, input_ids, attention_mask, token_type_ids = item
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
        
    
class data_generator_direct_rel(Dataset):
    def __init__(self, data, tokenizer, max_len, schema, add_neg=True, max_ngram=128, neg_num=128, label=0, mode='train'):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.schema = schema #spo
        self.max_ngram = max_ngram
        self.neg_num = neg_num 
        self.add_neg = add_neg
        self.labels = [label] * len(self.data)
        
        print(self.add_neg, "==apply add neg==")
        
        # self.features = []
        # for item in self.data:
        #     feature = self.encoder(item)
        #     if feature:
        #         self.features.append(feature)
        #     if len(self.features) == 1000 and mode == 'debug':
        #         break
                
        # print(self.features[0], '=====data=====', len(self.features))
                
        self.labels = [label] * len(self.data)
    
    def __len__(self):
        return len(self.data)

    def encoder(self, item):
        text = item["text"]
        encoder_text = self.tokenizer(text, return_offsets_mapping=True, max_length=self.max_len, truncation=True)
        input_ids = encoder_text["input_ids"]
        token_type_ids = encoder_text["token_type_ids"] #RoBERTa不需要NSP任务
        attention_mask = encoder_text["attention_mask"]
        spoes = set()
        spoes_rel = {}
        all_entity = set()

        for s, p, o, s_t, o_t in item["spo_list"]:
            if s and o:
                s = self.tokenizer.encode(s, add_special_tokens=False)
                p = self.schema[s_t + "_" + p + "_" +o_t]
                o = self.tokenizer.encode(o, add_special_tokens=False)
                sh_list = search_all(s, input_ids)
                oh_list = search_all(o, input_ids)
                if sh_list and oh_list:
                    for sh in sh_list:
                        for oh in oh_list:
                            spoes.add((sh, sh+len(s)-1, p, oh, oh+len(o)-1))
                            if (sh, sh+len(s)-1, oh, oh+len(o)-1) not in spoes_rel:
                                spoes_rel[(sh, sh+len(s)-1, oh, oh+len(o)-1)] = set()
                            spoes_rel[(sh, sh+len(s)-1, oh, oh+len(o)-1)].add(p)
                            all_entity.add((sh, sh+len(s)-1))
                            all_entity.add((oh, oh+len(o)-1))
        
        if self.add_neg:
            # token from https://github.com/LeePleased/NegSampling-NER/blob/master/model.py
            # negative sampling
            candies_all = flat_list([[(i, j) for j in range(i, min(self.max_ngram+i, len(input_ids)-1)) if (i, j) not in all_entity] for i in range(1, len(input_ids)-1)])
            random.shuffle(candies_all)
            
            if len(all_entity) < self.neg_num:
                neg_num = self.neg_num - len(all_entity)
                for cand in candies_all[:neg_num]:
                    all_entity.add(cand)
            
        # random.shuffle(all_entity)
                
        # get all candidiate entities
        all_entity = list(all_entity) #
        entity_size = len(all_entity) # neg_num, 2
        head_pos = [entity[0] for entity in all_entity] # neg_num of entity start-pos
        tail_pos = [entity[1] for entity in all_entity] # neg_num of entity end-pos
        
        # head_tail_labels = [set() for i in range(len(self.schema))]
        head_tail_labels = np.zeros((len(self.schema), self.neg_num, self.neg_num)).astype(np.float32)
        head_tail_mask = [1]*len(all_entity)
        
        for i, entity_s in enumerate(all_entity):
            for j, entity_o in enumerate(all_entity):
                if entity_s + entity_o in spoes_rel:
                    for p in spoes_rel[entity_s + entity_o]:
                        # head_tail_labels[p].add((i, j)) #类似TP-Linker     
                        head_tail_labels[p,i,j]=1.0
                        # print(p, '=====', self.tokenizer.decode(input_ids[entity_s[0]:entity_s[1]+1]), self.tokenizer.decode(input_ids[entity_o[0]:entity_o[1]+1]))
        
        # for label in head_tail_labels:
        #     if not label:
        #         label.add((0,0))

        # head_tail_labels = sequence_padding([list(l) for l in head_tail_labels])
        # print(head_tail_labels.shape)
        
        # if np.sum(head_tail_labels) == 0:
        #     print(item["spo_list"], '======', spoes)
        
        return text, head_pos, tail_pos, head_tail_labels, head_tail_mask, input_ids, attention_mask, token_type_ids

    def __getitem__(self, idx):
        # return self.features[idx]
        return self.encoder(self.data[idx])
    
    def get_labels(self):
        return self.labels

    @staticmethod
    def collate(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids = [], [], []
        batch_head_tail_labels, batch_head_tail_mask, batch_head_pos, batch_tail_pos = [], [], [], []
        text_list = []
        for item in examples:
            [text, head_pos, tail_pos, head_tail_labels, head_tail_mask, 
             input_ids, attention_mask, token_type_ids] = item
                        
            batch_head_pos.append(head_pos) # [batch_size, neg_num]
            batch_tail_pos.append(tail_pos) # [batch_size, neg_num]
            
            batch_head_tail_labels.append(head_tail_labels) # [batch_size,len(schema), neg_num, neg_num]
            batch_head_tail_mask.append(head_tail_mask) # [batch_size, neg_num, neg_num, len(schema)]
            
            batch_token_ids.append(input_ids) # [batch_size, seq_len]
            batch_mask_ids.append(attention_mask) # [batch_size, seq_len]
            batch_token_type_ids.append(token_type_ids) # [batch_size, seq_len]

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
                
        batch_head_pos = torch.tensor(sequence_padding(batch_head_pos)).long() # [batch_size, neg_num]
        batch_tail_pos = torch.tensor(sequence_padding(batch_tail_pos)).long() # [batch_size, neg_num]
        
        batch_head_tail_labels = torch.tensor(batch_head_tail_labels).long()
        batch_head_tail_mask = torch.tensor(sequence_padding(batch_head_tail_mask)).float() # [batch_size, num_head]
        
        return [text_list, batch_token_ids, batch_mask_ids, batch_token_type_ids, 
                batch_head_pos, batch_tail_pos, batch_head_tail_labels, batch_head_tail_mask]
    
    
class data_generator_negative_sampling(Dataset):
    def __init__(self, data, tokenizer, max_len, schema, label=0):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.schema = schema #spo
        self.labels = [label] * len(self.data)
        
    def __len__(self):
        return len(self.data)

    def encoder(self, item):
        text = item["text"]
        encoder_text = self.tokenizer(text, return_offsets_mapping=True, max_length=self.max_len, truncation=True)
        input_ids = encoder_text["input_ids"]
        token_type_ids = encoder_text["token_type_ids"] #RoBERTa不需要NSP任务
        attention_mask = encoder_text["attention_mask"]
        spoes = set()
        for s, p, o, s_t, o_t in item["spo_list"]:
            if s and o:
                s = self.tokenizer.encode(s, add_special_tokens=False)
                p = self.schema[s_t + "_" + p + "_" +o_t]
                o = self.tokenizer.encode(o, add_special_tokens=False)
                sh_list = search_all(s, input_ids)
                oh_list = search_all(o, input_ids)
                if sh_list and oh_list:
                    for sh in sh_list:
                        for oh in oh_list:
                            spoes.add((sh, sh+len(s)-1, p, oh, oh+len(o)-1))
        
        entity_labels = np.zeros((2, len(input_ids), len(input_ids)))
        head_labels = [set() for i in range(len(self.schema))]
        tail_labels = [set() for i in range(len(self.schema))]
        
        for sh, st, p, oh, ot in spoes:
            entity_labels[0, sh, st] = 1
            entity_labels[1, oh, ot] = 1
            head_labels[p].add((sh, oh)) #类似TP-Linker
            tail_labels[p].add((st, ot))
        for label in head_labels+tail_labels:
            if not label:
                label.add((0,0))
        # 例如entity = [{(1,3)}, {(4,5), (7,9)}]
        # entity[0]即{(1,3)}代表头实体首尾， entity[1]即{(4,5),{7,9}}代表尾实体首尾
        # 需要标签对齐为 [[[1,3][0,0]] , [[4,5][7,9]]]
        head_labels = sequence_padding([list(l) for l in head_labels])
        tail_labels = sequence_padding([list(l) for l in tail_labels])

        return text, entity_labels, head_labels, tail_labels, input_ids, attention_mask, token_type_ids

    def __getitem__(self, idx):
        item = self.data[idx]
        return self.encoder(item)
    
    def get_labels(self):
        return self.labels

    @staticmethod
    def collate(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids = [], [], []
        batch_entity_labels, batch_head_labels, batch_tail_labels = [], [], []
        text_list = []
        
        max_len = 0
        for item in examples:
            text, entity_labels, head_labels, tail_labels, input_ids, attention_mask, token_type_ids = item
            if max_len < entity_labels.shape[-1]:
                max_len = entity_labels.shape[-1]
        
        for item in examples:
            text, entity_labels, head_labels, tail_labels, input_ids, attention_mask, token_type_ids = item
            
            entity_labels_ = np.zeros((2, max_len, max_len))
            lens = entity_labels.shape[-1]
            entity_labels_[:, 0:lens, 0:lens] = entity_labels
            
            batch_entity_labels.append(entity_labels_) # batch_size, 2, seq_len, seq_len
            batch_head_labels.append(head_labels)
            batch_tail_labels.append(tail_labels)
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            text_list.append(text)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_entity_labels = torch.tensor(batch_entity_labels).long()
        batch_head_labels = torch.tensor(sequence_padding(batch_head_labels, seq_dims=2)).long()
        batch_tail_labels = torch.tensor(sequence_padding(batch_tail_labels, seq_dims=2)).long()\
        
        return text_list, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels, batch_head_labels, batch_tail_labels