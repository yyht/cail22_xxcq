import json
import numpy as np
import torch
from torch.utils.data import Dataset
from operator import is_not
from functools import partial
import random

def load_name(filename):
    """
    {"text": "对儿童SARST细胞亚群的研究表明，与成人SARS相比，儿童细胞下降不明显，证明上述推测成立。", "entity_list": [{"start_idx": 3, "end_idx": 9, "type": "身体部位", "entity": "SARST细胞"}, {"start_idx": 19, "end_idx": 24, "type": "疾病", "entity": "成人SARS"}]}
    """
    D = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = json.loads(line)
            entity_list = []
            for entity in line["entity_list"]:
                entity_list.append((entity['entity'], entity['type'], entity['start_idx']))
            D.append({
                "text":line["text"],
                "entity_list":entity_list
            })
        random.shuffle(D)
        return D

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
            label_mask.append(1.0)
        elif dt_type in ['general_entity', 'medical_entity', 'conv_medical_entity', 'yiduyun_medical_entity']:
            entity_labels = [set() for i in range(len(schema_dict[dt_type]['schema']))]
            for label in entity_labels:
                if not label:
                    label.add((0,0))
            entity_labels = sequence_padding([list(l) for l in entity_labels])
            label_mask.append(0.0)
        else:
            continue
        multitask_labels_list.append([entity_labels])
    return multitask_labels_list, label_mask

class data_generator_entity(Dataset):
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
        
        print(self.data_type, '==data_type==', self.schema, '====', self.schema_dict)
        
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
        entities = set()
        for entity, entity_type, entity_start_idx in item["entity_list"]:
            entity_label = self.schema[entity_type]
            if entity_start_idx != -1:
                start, end = entity_start_idx, entity_start_idx + len(entity) - 1
                start_t, end_t = char_span_to_token_span(char2token, (start, end+1))
                if input_ids[start_t:end_t]:
                    entities.add((entity_label, start_t, end_t-1))
            else:
                entity_ids = self.tokenizer.encode(entity, add_special_tokens=False)
                a_list = search_all(entity_ids, input_ids)
                if a_list:
                    for a_i in a_list:
                        start, end = a_i, a_i + len(entity) - 1
                        entities.add((entity_label, start, end))
        # 构建标签
        entity_labels = [set() for i in range(len(self.schema))]
        
        for label, start, end in entities:
            entity_labels[label].add((start, end))
        
        for label in entity_labels:
            if not label:
                label.add((0,0))
        # 例如entity = [{(1,3)}, {(4,5), (7,9)}]
        # entity[0]即{(1,3)}代表头实体首尾， entity[1]即{(4,5),{7,9}}代表尾实体首尾
        # 需要标签对齐为 [[[1,3][0,0]] , [[4,5][7,9]]]
        entity_labels = sequence_padding([list(l) for l in entity_labels])
        
        multitask_labels_list, label_mask = multitask_labels([entity_labels], self.data_type, self.schema_dict)
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
        
        for item in examples: # batch_size
            input_ids, attention_mask, token_type_ids, multitask_labels_list, label_mask = item
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            batch_task_mask.append(label_mask)
            
            for idx, (multitask_label, task_label) in enumerate(zip(multitask_labels_list, label_mask)):
                batch_argu_labels[idx].append(multitask_label[0])

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_task_mask = torch.tensor(sequence_padding(batch_task_mask)).float()#RoBERTa 不需要NSP
        
        for idx, item in enumerate(batch_argu_labels):
            batch_argu_labels[idx] = torch.tensor(sequence_padding(batch_argu_labels[idx], seq_dims=2)).long()
        
        return batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_task_mask, batch_argu_labels
    
