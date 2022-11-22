import json
import numpy as np
import torch
from torch.utils.data import Dataset
from operator import is_not
from functools import partial
import random

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

def search_all(pattern, sequence):
    all_index = []
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            all_index.append(i)
    return all_index

def multitask_labels(dt_labels, data_type, schema_dict):
    
    multitask_labels_list = []
    label_mask = []
    
    dt_list = list(schema_dict.keys())
    for dt_type in dt_list:
        if dt_type == data_type:
            entity_labels = dt_labels[0]
            head_labels = dt_labels[1]
            tail_labels = dt_labels[2]
            label_mask.append(1.0)
        elif dt_type in ['duie_life', 'asa', 'duie_org', 'asa_medical', 
                         'duie_fin_monitor', 'duie_cblue_mceie', 'duie_car_information']:
            entity_labels = [set() for i in range(2)]
            head_labels = [set() for i in range(len(schema_dict[dt_type]['schema']))]
            tail_labels = [set() for i in range(len(schema_dict[dt_type]['schema']))]
            
            for label in entity_labels+head_labels+tail_labels:
                if not label:
                    label.add((0,0))
            entity_labels = sequence_padding([list(l) for l in entity_labels])
            head_labels = sequence_padding([list(l) for l in head_labels])
            tail_labels = sequence_padding([list(l) for l in tail_labels])
            label_mask.append(0.0)
        elif dt_type in ['duee', 'duee_dieaese', 'duee_fin_news', 'duee_fin', 'duee_fewfc_2022']:
            entity_labels = [set() for _ in range(len(schema_dict[dt_type]['schema']))]
            head_labels, tail_labels = set(), set()
            
            for label in entity_labels + [head_labels, tail_labels]:
                if not label:  # 至少要有一个标签
                    label.add((0, 0))  # 如果没有则用0填充
            entity_labels = sequence_padding([list(l) for l in entity_labels])
            head_labels = sequence_padding([list(head_labels)])
            tail_labels = sequence_padding([list(tail_labels)])
            label_mask.append(0.0)
        else:
            continue
        multitask_labels_list.append([entity_labels, head_labels, tail_labels])
    return multitask_labels_list, label_mask

class data_generator_duee(Dataset):
    def __init__(self, data, tokenizer, max_len, schema_dict, data_type, label=0, greedy_search=False, seg_token='<S>'):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data_type = data_type
        self.schema_dict = schema_dict
        self.schema = self.schema_dict[self.data_type]['schema'] #spo (event_type, role_type)
        self.task_num = len(self.schema_dict)
        self.labels = [label] * len(self.data)
        self.greedy_search = greedy_search
        
        self.seg_token = seg_token
        
        self.tokenizer.add_special_tokens({
                    "additional_special_tokens": [self.seg_token] })
        
    def __len__(self):
        return len(self.data)

    def encoder(self, item):
        text = item["text"]
        
        encoder_text = self.tokenizer(text, return_offsets_mapping=True, max_length=self.max_len, truncation=True)
        input_ids = encoder_text["input_ids"]
        
        token2char = encoder_text.offset_mapping
        char2token = [None] * len(text)
        for i, ((start, end)) in enumerate(token2char):
            char2token[start:end] = [i] * (end - start)
        
        input_ids = encoder_text["input_ids"]
        token_type_ids = encoder_text["token_type_ids"] #RoBERTa不需要NSP任务
        attention_mask = encoder_text["attention_mask"]
        events = []
        for e in item['events']:
            events.append([])
            for e, r, a, i in e:
                label = self.schema[(e, r)]
                if i != -1:
                    start, end = i, i + len(a) - 1
                    start_t, end_t = char_span_to_token_span(char2token, (start, end+1))
                    if input_ids[start_t:end_t]:
                        events[-1].append((label, start_t, end_t-1))
                else:
                    if self.greedy_search:
                        argument_ids = self.tokenizer.encode(a, add_special_tokens=False)
                        a_list = search_all(argument_ids, input_ids)
                        if a_list:
                            for a_i in a_list:
                                start, end = a_i, a_i + len(argument_ids) - 1
                                events[-1].append((label, start, end))
        # 构建标签
        argu_labels = [set() for _ in range(len(self.schema))]
        head_labels, tail_labels = set(), set()
        for e in events:
            for l, h, t in e:
                argu_labels[l].add((h, t))
            for i1, (_, h1, t1) in enumerate(e):
                for i2, (_, h2, t2) in enumerate(e):
                    if i2 > i1:
                        head_labels.add((min(h1, h2), max(h1, h2)))
                        tail_labels.add((min(t1, t2), max(t1, t2)))
        for label in argu_labels + [head_labels, tail_labels]:
            if not label:  # 至少要有一个标签
                label.add((0, 0))  # 如果没有则用0填充

        argu_labels = sequence_padding([list(l) for l in argu_labels])
        head_labels = sequence_padding([list(head_labels)])
        tail_labels = sequence_padding([list(tail_labels)])
        
        multitask_labels_list, label_mask = multitask_labels([argu_labels, head_labels, tail_labels], self.data_type, self.schema_dict)
        return input_ids, attention_mask, token_type_ids, multitask_labels_list, label_mask

    def __getitem__(self, idx):
        item = self.data[idx]
        return self.encoder(item)
    
    def get_labels(self):
        return self.labels

    @staticmethod
    def collate(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids = [], [], []
        batch_task_mask = []
        input_ids, attention_mask, token_type_ids, multitask_labels_list, label_mask = examples[0]
        task_num = len(multitask_labels_list)
        
        batch_argu_labels =  [list() for i in range(task_num)]
        batch_head_labels = [list() for i in range(task_num)]
        batch_tail_labels = [list() for i in range(task_num)]
        
        for item in examples: # batch_size
            input_ids, attention_mask, token_type_ids, multitask_labels_list, label_mask = item
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            batch_task_mask.append(label_mask)
            
            for idx, (multitask_label, task_label) in enumerate(zip(multitask_labels_list, label_mask)):
                batch_argu_labels[idx].append(multitask_label[0])
                batch_head_labels[idx].append(multitask_label[1])
                batch_tail_labels[idx].append(multitask_label[2])

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_task_mask = torch.tensor(sequence_padding(batch_task_mask)).float()#RoBERTa 不需要NSP
        
        for idx, item in enumerate(batch_argu_labels):
            batch_argu_labels[idx] = torch.tensor(sequence_padding(batch_argu_labels[idx], seq_dims=2)).long()
            
        for idx, item in enumerate(batch_head_labels):
            batch_head_labels[idx] = torch.tensor(sequence_padding(batch_head_labels[idx], seq_dims=2)).long()
            
        for idx, item in enumerate(batch_tail_labels):
            batch_tail_labels[idx] = torch.tensor(sequence_padding(batch_tail_labels[idx], seq_dims=2)).long()
        
        return batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_task_mask, batch_argu_labels, batch_head_labels, batch_tail_labels
    
class data_generator_duie(Dataset):
    def __init__(self, data, tokenizer, max_len, schema_dict, data_type, label=0, greedy_search=False, seg_token='<S>'):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data_type = data_type
        self.schema_dict = schema_dict
        self.schema = self.schema_dict[self.data_type]['schema'] #spo (event_type, role_type)
        self.task_num = len(self.schema_dict)
        self.labels = [label] * len(self.data)
        self.greedy_search = greedy_search
        
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
        
        multitask_labels_list, label_mask = multitask_labels([entity_labels, head_labels, tail_labels], self.data_type, self.schema_dict)
        return input_ids, attention_mask, token_type_ids, multitask_labels_list, label_mask

    def __getitem__(self, idx):
        item = self.data[idx]
        return self.encoder(item)
    
    def get_labels(self):
        return self.labels

    @staticmethod
    def collate(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids = [], [], []
        batch_task_mask = []
        input_ids, attention_mask, token_type_ids, multitask_labels_list, label_mask = examples[0]
        task_num = len(multitask_labels_list)
        
        batch_argu_labels =  [list() for i in range(task_num)]
        batch_head_labels = [list() for i in range(task_num)]
        batch_tail_labels = [list() for i in range(task_num)]
        
        for item in examples:
            input_ids, attention_mask, token_type_ids, multitask_labels_list, label_mask = item
                        
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            batch_task_mask.append(label_mask)
            
            for idx, (multitask_label, task_label) in enumerate(zip(multitask_labels_list, label_mask)):
                batch_argu_labels[idx].append(multitask_label[0])
                batch_head_labels[idx].append(multitask_label[1])
                batch_tail_labels[idx].append(multitask_label[2])

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_task_mask = torch.tensor(sequence_padding(batch_task_mask)).float()#RoBERTa 不需要NSP
        
        for idx, item in enumerate(batch_argu_labels):
            batch_argu_labels[idx] = torch.tensor(sequence_padding(batch_argu_labels[idx], seq_dims=2)).long()
            
        for idx, item in enumerate(batch_head_labels):
            batch_head_labels[idx] = torch.tensor(sequence_padding(batch_head_labels[idx], seq_dims=2)).long()
            
        for idx, item in enumerate(batch_tail_labels):
            batch_tail_labels[idx] = torch.tensor(sequence_padding(batch_tail_labels[idx], seq_dims=2)).long()
        
        return batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_task_mask, batch_argu_labels, batch_head_labels, batch_tail_labels