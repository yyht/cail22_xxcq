
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
from utils.augment import insert_punctuation_marks

from utils.seq2struct_dataloader_online import data_generator_single_schema
from utils.seq2struct_dataloader import (deleteDuplicate_v1, char_span_to_token_span, 
                                         token_span_to_char_span, get_token2char_char2token, 
                                         sequence_padding, search_all, search)



class data_generator_single_schema(Dataset):
    def __init__(self, data, tokenizer, max_len, schema, label=0, 
                 task_dict={}, mode='train', build_data=True, add_neg=False, add_role_shuffle=False,
                doc_stride=32, offset=0):
        self.data = data
        self.doc_stride = doc_stride
        self.offset = offset
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.task_dict = task_dict
        self.sep_token = task_dict['sep_token']
        self.seg_token = task_dict['seg_token']
        self.group_token = task_dict['group_token']
        self.start_token = task_dict['start_token']
        self.end_token = task_dict['end_token']
        self.sentinel_token = task_dict['sentinel_token']
        self.sentinel_start_idx = task_dict['sentinel_start_idx']
        self.greedy_search = self.task_dict.get('greedy_search', False)
        self.mode = mode
        self.build_data = build_data
        self.add_neg = self.task_dict.get('add_neg', False)
        self.add_role_shuffle = self.task_dict.get('role_shuffle', False)
        self.schema_shuffle = self.task_dict.get('schema_shuffle', False)
        self.role_schema_order = self.task_dict.get('role_schema_order', False)
        self.remove_dup = self.task_dict.get('remove_dup', False)
        self.add_spefical_tokens = self.task_dict.get('add_spefical_tokens', True)
        self.search_mode = self.task_dict.get('search_mode', 'token_id')
        
        if self.add_spefical_tokens:
            self.tokenizer.add_special_tokens({
                    "additional_special_tokens": [ self.sentinel_token.format(i) for i in range(1, 100)]+[self.seg_token, self.sep_token, 
                                                                                                self.group_token, self.start_token, self.end_token]})
        
        self.schema_dict = {}
        for schema_dict in schema:
            if schema_dict['type'] not in self.schema_dict:
                self.schema_dict[schema_dict['type']] = {
                        'role2sentinel':{},
                        'sentinel2role':{},
                        'role2type':{},
                        'type2role':{}
                }
            for role_index, role_dict in enumerate(schema_dict['role_list']):
                role_type = role_dict['type'] + role_dict['role']
                self.schema_dict[schema_dict['type']]['role2sentinel'][role_type] = self.sentinel_token.format(role_index+self.sentinel_start_idx)
                self.schema_dict[schema_dict['type']]['sentinel2role'][self.sentinel_token.format(role_index+self.sentinel_start_idx)] = role_type
                self.schema_dict[schema_dict['type']]['role2type'][role_dict['role']] = role_type
                self.schema_dict[schema_dict['type']]['type2role'][role_type] = role_dict['role']
        
        self.features = []
        for item in self.data:
            total_target_set = set()
            for target_dict in item['target_list']:
                target_type = target_dict['type']
                total_target_set.add(target_type)
            
            negative_type_list = [target_type for target_type in self.self.schema_dict if target_type not in list(total_target_set)]
            np.random.permutation(negative_type_list)[0:3]
            
            input_ids = self.tokenizer(item['text'], return_offsets_mapping=True, add_special_tokens=False)['input_ids']
            doc_spans = slide_window(input_ids, self.max_len, self.doc_stride, offset=self.offset)
            for doc_span in doc_spans:
                span_start = doc_span.start
                span_end = doc_span.start + doc_span.length
            
                for target_type in list(total_target_set):
                    content = {}
                    for key in item:
                        content[key] = item[key]
                    content['span_start'] = span_start
                    content['span_end'] = span_end
                    content['target_type'] = target_type
                    content['label'] = 1
                    self.features.append(content)
                    
                for target_negative_type in negative_type_list:
                    content = {}
                    for key in item:
                        content[key] = item[key]
                    content['span_start'] = span_start
                    content['span_end'] = span_end
                    content['target_type'] = target_negative_type
                    content['label'] = 0
                    self.features.append(content)
        
        import random
        random.shuffle(self.features)
        self.labels = [label] * len(self.features)
        self._task_id = label
                
    def __len__(self):
        return len(self.features)

    def encoder(self, item):
        text = item["text"]
        
        instruction_text = self.start_token + self.task_dict['instruction'] + self.sep_token
        encoder_instruction_text = self.tokenizer(instruction_text, return_offsets_mapping=True, max_length=self.max_len, truncation=True, add_special_tokens=False)
        instruction_input_ids = encoder_instruction_text["input_ids"]
        instruction_token_type_ids = encoder_instruction_text.get("token_type_ids", []) #RoBERTa不需要NSP任务
        instruction_attention_mask = encoder_instruction_text["attention_mask"] #RoBERTa不需要NSP任务
        
        encoder_text = self.tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        input_ids = encoder_text["input_ids"]
        token_type_ids = encoder_text.get("token_type_ids", []) #RoBERTa不需要NSP任务
        attention_mask = encoder_text["attention_mask"]
        
        token2char = encoder_text.offset_mapping
        char2token = [None] * len(text)
        for i, ((start, end)) in enumerate(token2char):
            char2token[start:end] = [i] * (end - start)
            
        span_start = item['span_start']
        span_end = item['span_end']

        span_input_ids = input_ids[span_start:span_end] + self.tokenizer(self.sep_token, add_special_tokens=False)['input_ids']
        span_type_ids = [0] * len(span_input_ids)
        span_attention_mask = [1] * len(span_input_ids)
        
        output_list = []
        
        target_type = item['target_type']
        label = item['label']
        
        schema_strings = target_type + self.sep_token
        encoder_schema_text = self.tokenizer(schema_strings, return_offsets_mapping=True, max_length=self.max_len, truncation=False, add_special_tokens=False)
        schema_input_ids = encoder_schema_text["input_ids"]
        schema_token_type_ids = encoder_schema_text.get("token_type_ids", []) #RoBERTa不需要NSP任务
        schema_attention_mask = encoder_schema_text["attention_mask"]
            
        if self.task_dict.get('augment_marks', False):
            import random
            if random.random() > 0.3:
                span_sentence = " ".join(self.tokenizer.tokenize(self.tokenizer.decode(span_input_ids)))
                span_sentence = insert_punctuation_marks(span_sentence)
                span_input_ids = self.tokenizer(span_sentence, add_special_tokens=False)['input_ids']
                span_type_ids = [0] * len(span_input_ids)
                span_attention_mask = [1] * len(span_input_ids)
                    
        output_list.append((text, instruction_input_ids, instruction_token_type_ids, instruction_attention_mask,
           span_input_ids, span_type_ids, span_attention_mask,
           schema_input_ids, schema_token_type_ids, schema_attention_mask,
           label))
            
        return output_list
    
    def __getitem__(self, idx):
        return self.encoder(self.features[idx])[0]
    
    def get_labels(self):
        return self.labels
    
    def get_task_id(self):
        return self._task_id

    @staticmethod
    def collate_nli(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids = [], [], []
        batch_label_type = []
        for item in examples:
            (text, instruction_input_ids, instruction_token_type_ids, instruction_attention_mask,
            input_ids, token_type_ids, attention_mask,
            schema_input_ids, schema_token_type_ids, schema_attention_mask, 
            label) = item
            
            # cls instruction sep input sep schema sep target end
            input_ids = instruction_input_ids + input_ids + schema_input_ids
            attention_mask = [1] * len(input_ids)
            token_type_ids = [0] * len(input_ids)
            
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            batch_label_type.append(label)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_label_type = torch.tensor(batch_label_type).long()
        
        return batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_label_type