
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

from utils.seq2struct_dataloader_online import data_generator_single_schema
from utils.seq2struct_dataloader import (deleteDuplicate_v1, char_span_to_token_span, 
                                         token_span_to_char_span, get_token2char_char2token, 
                                         sequence_padding, search_all, search)



class data_generator_single_schema(data_generator_single_schema):
    def __init__(self, data, tokenizer, max_len, schema, label=0, 
                 task_dict={}, mode='train', build_data=True, add_neg=False, add_role_shuffle=False,
                doc_stride=32, offset=0):
        
        super().__init__(data, tokenizer, max_len, schema, label, 
                 task_dict, mode, build_data, add_neg, add_role_shuffle,
                doc_stride, offset)
        
    
    def encoder(self, item):
        text = item["text"]
        
        flag = False
        if text == "长城汽车上涨3% 上周四及周五获董事长增持\n客户端\n新浪港股讯，长城汽车(5.22,0.09,1.75%)（02333）H股现价升3.05%，报5.06元，盘中高见5.12元；成交约845万股，涉资4273万元。\nA股（沪：601633）现价8.1元人民币，升0.11元人民币，或升1.38%，成交1993万元人民币，涉及246万股．现价A股对H股呈溢价+74%。":
            print(item, '==========')
            flag = False
        
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
        
        total_target_dict = {}
        for target_dict in item['target_list']:
            target_type = target_dict['type']
            if target_type not in total_target_dict:
                total_target_dict[target_type] = []
            if self.remove_dup:
                role_list = deleteDuplicate_v1(target_dict['role_list'])
                role_list = sorted(role_list, key=lambda item:item['argument'])
            else:
                role_list = target_dict['role_list']
            total_target_dict[target_type].append(role_list)
            
        if flag:
            print(total_target_dict, '=========before========')
        
        for target_type in total_target_dict:
            before_num = len(total_target_dict[target_type])
            if before_num >= 2:
                if self.remove_dup:
                    total_target_dict[target_type] = deleteDuplicate_v1(total_target_dict[target_type])
                after_num = len(total_target_dict[target_type])
            
        if flag:
            print(total_target_dict, '========after=========')
            
        if total_target_dict:
            assert len(total_target_dict) == 1
            
        span_start = item['span_start']
        span_end = item['span_end']

        span_input_ids = input_ids[span_start:span_end] + self.tokenizer(self.sep_token, add_special_tokens=False)['input_ids']
        span_type_ids = [0] * len(span_input_ids)
        span_attention_mask = [1] * len(span_input_ids)
        
        output_list = []
        
        for target_type in total_target_dict:
            schema_dict = self.schema_dict[target_type]
            if self.task_dict['add_schema_type']:
                schema_strings = target_type + self.sep_token
            else:
                schema_strings = ''
            key_list = list(schema_dict['role2sentinel'])
            if self.schema_shuffle:
                import random
                if random.random() > 0.5:
                    random.shuffle(key_list)

            for role in key_list:
                schema_strings += role + schema_dict['role2sentinel'][role] # role sentinel

            schema_strings += self.sep_token
            encoder_schema_text = self.tokenizer(schema_strings, return_offsets_mapping=True, max_length=self.max_len, truncation=False, add_special_tokens=False)
            schema_input_ids = encoder_schema_text["input_ids"]
            schema_token_type_ids = encoder_schema_text.get("token_type_ids", []) #RoBERTa不需要NSP任务
            schema_attention_mask = encoder_schema_text["attention_mask"]

            target_strings = ''
            if self.add_role_shuffle:
                import random
                if random.random() > 0.5:
                    random.shuffle(total_target_dict[target_type])
            for role_list in total_target_dict[target_type]:
                if self.role_schema_order:
                    role_index_list = []
                    for key_index, key in enumerate(key_list):
                        for role_index, role_dict in enumerate(role_list):
                            if key == role_dict['type'] + role_dict['role']:
                                role_index_list.append(role_index)
                else:
                    role_index_list = range(len(role_list))

                target_dict = OrderedDict({})
                for role_index in role_index_list:
                    role_dict = role_list[role_index]
                    argument_start_index = role_dict.get('argument_start_index', -1)
                    role_type = role_dict['type'] + role_dict['role']
                    argument = role_dict['argument']
                    if argument_start_index != -1:
                        start, end = argument_start_index, argument_start_index + len(role_dict['argument']) - 1
                        start_t, end_t = char_span_to_token_span(char2token, (start, end+1))
                        if input_ids[start_t:end_t] and start_t >= span_start and end_t <= span_end:
                            if (argument, schema_dict['role2sentinel'][role_type]) not in target_dict:
                                target_dict[(argument, schema_dict['role2sentinel'][role_type])] = ''
                        else:
                            if self.greedy_search:
                                if self.search_mode == 'token_id':
                                    arguemnt_ids = self.tokenizer.encode(argument, add_special_tokens=False)
                                    sh = search(arguemnt_ids, span_input_ids)
                                elif self.search_mode == 'string':
                                    span_text_pos = token_span_to_char_span(token2char, [span_start, span_end])
                                    span_text = text[span_text_pos[0]:span_text_pos[1]]
                                    sh = search(argument, span_text)
                                if sh != -1:
                                    # target_strings += argument + schema_dict['role2sentinel'][role_type]
                                    if (argument, schema_dict['role2sentinel'][role_type]) not in target_dict:
                                        target_dict[(argument, schema_dict['role2sentinel'][role_type])] = ''
                                    
                    else:
                        if self.search_mode == 'token_id':
                            arguemnt_ids = self.tokenizer.encode(argument, add_special_tokens=False)
                            sh = search(arguemnt_ids, span_input_ids)
                        elif self.search_mode == 'string':
                            span_text_pos = token_span_to_char_span(token2char, [span_start, span_end])
                            span_text = text[span_text_pos[0]:span_text_pos[1]]
                            sh = search(argument, span_text)
                        if sh != -1:
                            # target_strings += argument + schema_dict['role2sentinel'][role_type]
                            if (argument, schema_dict['role2sentinel'][role_type]) not in target_dict:
                                target_dict[(argument, schema_dict['role2sentinel'][role_type])] = ''

                add_group_flag = False
                for key_set in target_dict:
                    target_strings += "".join(list(key_set))
                    add_group_flag = True
                if flag:
                    print(target_dict, '=====target_dict====')

                if add_group_flag:
                    target_strings += self.group_token

            target_strings += self.end_token
            encoder_target_text = self.tokenizer(target_strings, return_offsets_mapping=True, max_length=self.max_len, truncation=False, add_special_tokens=False)
            target_input_ids = encoder_target_text["input_ids"]
            target_token_type_ids = encoder_target_text.get("token_type_ids", []) #RoBERTa不需要NSP任务
            target_attention_mask = encoder_target_text["attention_mask"]
            
            if self.task_dict.get('augment_marks', False):
                import random
                if random.random() > 0.5:
                    span_sentence = " ".join(self.tokenizer.tokenize(self.tokenizer.decode(span_input_ids)))
                    span_sentence = insert_punctuation_marks(span_sentence)
                    span_input_ids = self.tokenizer(span_sentence, add_special_tokens=False)['input_ids']
                    span_type_ids = [0] * len(span_input_ids)
                    span_attention_mask = [1] * len(span_input_ids)
                    
            output_list.append((text, instruction_input_ids, instruction_token_type_ids, instruction_attention_mask,
               span_input_ids, span_type_ids, span_attention_mask,
               schema_input_ids, schema_token_type_ids, schema_attention_mask,
               target_input_ids, target_token_type_ids, target_attention_mask))
            
        return output_list
    
    @staticmethod
    def collate_unilm(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids = [], [], []
        batch_loss_ratio = []
        batch_loss_mask = []
        for item in examples:
            (text, instruction_input_ids, instruction_token_type_ids, instruction_attention_mask,
               input_ids, token_type_ids, attention_mask,
               schema_input_ids, schema_token_type_ids, schema_attention_mask,
               target_input_ids, target_token_type_ids, target_attention_mask) = item
            
            import random
            if random.random() > 0.5:
                target_prefix_pos = random.randint(0, len(target_input_ids)-1)
            else:
                target_prefix_pos = 0
            target_prefix_input_ids = target_input_ids[:target_prefix_pos]
            target_suffix_input_ids = target_input_ids[target_prefix_pos:]
            
            # cls instruction sep input sep schema sep target end
            input_ids = instruction_input_ids + input_ids + schema_input_ids
            attention_mask = [1] * len(input_ids)
            token_type_ids = [0] * len(input_ids)
            loss_mask = [0] * len(input_ids)
            
            input_ids += target_input_ids
            attention_mask += [1] * len(target_input_ids)
            token_type_ids += [1] * len(target_input_ids)
            loss_mask += [0] * len(target_prefix_input_ids) + [1] * len(target_suffix_input_ids)
            
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            batch_loss_mask.append(loss_mask)
            batch_loss_ratio.append(len(target_input_ids)/(len(target_input_ids)-target_prefix_pos))

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_loss_ratio = torch.tensor(batch_loss_ratio).float()
        batch_loss_mask = torch.tensor(sequence_padding(batch_loss_mask)).float()#RoBERTa 不需要NSP
        
        return batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_loss_ratio, batch_loss_mask