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

from utils.augment import insert_punctuation_marks
import random


"""
@Auth: Xhw
@Description: CHIP/CBLUE 医学实体关系抽取，数据来源 https://tianchi.aliyun.com/dataset/dataDetail?dataId=95414
"""
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import random

from utils.seq2struct_dataloader_online import data_generator_single_schema, data_generator_flatten_schema
from utils.seq2struct_dataloader import (deleteDuplicate_v1, char_span_to_token_span, 
                                         token_span_to_char_span, get_token2char_char2token, 
                                         sequence_padding, search_all, search)

def load_ie_cond_schema(filename):
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
                'role_list':[{'role':line['condition_type'], 'type':''}, {'role':line['subject_type'], 'type':''}]
            }
            for key in line['object_type']:
                schema_dict['role_list'].append({'role':line['object_type'][key], 'type':''})
            D.append(schema_dict)
    print(D, '===schema===')
    return D

def load_chip_casual(filename):
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
                event_dict['role_list'].append({'role':spo['condition_type'], 'argument':spo['condition'], 'type':'', 'argument_start_index':-1})
                event_dict['role_list'].append({'role':spo['subject_type'], 'argument':spo['subject'], 'type':'', 'argument_start_index':-1})
                for key in spo['object_type']:
                    event_dict['role_list'].append({'role':spo['object_type'][key], 'argument':spo['object'][key], 'argument_start_index':-1, 'type':''})
                target_list.append(event_dict)
            D.append({
                "text":line["text"],
                "target_list":target_list
            })
        random.shuffle(D)
        print(filename, '=====size of data=====', len(D))
        print(D[0])
        return D

def load_fin_casual(filename):
    """
    {"text": "消失的“外企光环”，5月份在华裁员900余人，香饽饽变“臭”了", "id": "cba11b5059495e635b4f95e7484b2684", "event_list": [{"event_type": "组织关系-裁员", "trigger": "裁员", "trigger_start_index": 15, "arguments": [{"argument_start_index": 17, "role": "裁员人数", "argument": "900余人", "alias": []}, {"argument_start_index": 10, "role": "时间", "argument": "5月份", "alias": []}], "class": "组织关系"}]}
    """
    D = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = json.loads(line.strip())
            d = {'text': line['text'], 'target_list': [], 'cond':line['cond'], 'id':line['id']}
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

class data_generator_single_schema_ccks_casual(data_generator_single_schema):
    def __init__(self, data, tokenizer, max_len, schema, label=0, 
                 task_dict={}, mode='train', build_data=True, add_neg=False, add_role_shuffle=False,
                doc_stride=32, offset=0):
        
        super().__init__(data, tokenizer, max_len, schema, label, 
                 task_dict, mode, build_data, add_neg, add_role_shuffle,
                doc_stride, offset)
        
    def encoder(self, item):
        text = item["text"]
        
        import random
        if 'candidate_type' in item:
            random.shuffle(item['candidate_type'])
            event_dict = {
                            'type':item['candidate_type'][0],
                            'role_list':[]
                        }
            item['target_list'] = [event_dict]
        
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
                # print(total_target_dict[target_type], '=========before========')
                if self.remove_dup:
                    total_target_dict[target_type] = deleteDuplicate_v1(total_target_dict[target_type])
                after_num = len(total_target_dict[target_type])
                # if before_num != after_num:
                #     print(total_target_dict[target_type], '=========after========')
            
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
            schema_strings = ''
            key_list = list(schema_dict['role2sentinel'])
            if self.schema_shuffle:
                import random
                if random.random() > 0.5:
                    random.shuffle(key_list)
                    
            for cond in item.get('cond', []):
                for key in cond:
                    schema_strings += key + self.sep_token + cond[key] + self.sep_token
            
            if self.task_dict['add_schema_type']:
                schema_strings += target_type + self.sep_token
                
            for role in key_list:
                schema_strings += role + schema_dict['role2sentinel'][role] # role sentinel

            schema_strings += self.sep_token
            if self.task_dict.get('add_cardinality', False):
                schema_strings += str(len(total_target_dict[target_type])) + self.sep_token
            
            encoder_schema_text = self.tokenizer(schema_strings, return_offsets_mapping=True, add_special_tokens=False)
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
                    if argument:
                        if argument_start_index != -1:
                            start, end = argument_start_index, argument_start_index + len(role_dict['argument']) - 1
                            start_t, end_t = char_span_to_token_span(char2token, (start, end+1))
                            argument_ids = self.tokenizer.encode(argument, add_special_tokens=False)
                            if input_ids[start_t:end_t] and argument_ids and start_t >= span_start and end_t <= span_end:
                                if (argument, schema_dict['role2sentinel'][role_type]) not in target_dict:
                                    target_dict[(argument, schema_dict['role2sentinel'][role_type])] = ''
                            else:
                                if self.greedy_search:
                                    argument_ids = self.tokenizer.encode(argument, add_special_tokens=False)
                                    if self.search_mode == 'token_id':
                                        argument_ids = self.tokenizer.encode(argument, add_special_tokens=False)
                                        sh = search(argument_ids, span_input_ids)
                                    elif self.search_mode == 'string':
                                        span_text_pos = token_span_to_char_span(token2char, [span_start, span_end])
                                        span_text = text[span_text_pos[0]:span_text_pos[1]]
                                        sh = search(argument, span_text)
                                    if sh != -1 and argument_ids:
                                        # target_strings += argument + schema_dict['role2sentinel'][role_type]
                                        if (argument, schema_dict['role2sentinel'][role_type]) not in target_dict:
                                            target_dict[(argument, schema_dict['role2sentinel'][role_type])] = ''

                        else:
                            argument_ids = self.tokenizer.encode(argument, add_special_tokens=False)
                            if self.search_mode == 'token_id':
                                argument_ids = self.tokenizer.encode(argument, add_special_tokens=False)
                                sh = search(argument_ids, span_input_ids)
                            elif self.search_mode == 'string':
                                span_text_pos = token_span_to_char_span(token2char, [span_start, span_end])
                                span_text = text[span_text_pos[0]:span_text_pos[1]]
                                sh = search(argument, span_text)
                            if sh != -1 and argument_ids:
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
                    
            # if item['id'] in ['8638_1']:
            #     print(target_strings, target_dict, role_list)
            #     print(self.tokenizer.decode(span_input_ids))

            target_strings += self.end_token
            encoder_target_text = self.tokenizer(target_strings, return_offsets_mapping=True, max_length=self.max_len, truncation=False, add_special_tokens=False)
            target_input_ids = encoder_target_text["input_ids"]
            target_token_type_ids = encoder_target_text.get("token_type_ids", []) #RoBERTa不需要NSP任务
            target_attention_mask = encoder_target_text["attention_mask"]
            
            if self.task_dict.get('augment_marks', False):
                import random
                if random.random() > 0.4:
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
        for item in examples:
            (text, instruction_input_ids, instruction_token_type_ids, instruction_attention_mask,
               input_ids, token_type_ids, attention_mask,
               schema_input_ids, schema_token_type_ids, schema_attention_mask,
               target_input_ids, target_token_type_ids, target_attention_mask) = item
            
            # cls instruction sep input sep schema sep target end
            input_ids = instruction_input_ids + input_ids + schema_input_ids
            attention_mask = [1] * len(input_ids)
            token_type_ids = [0] * len(input_ids)
            
            input_ids += target_input_ids
            attention_mask += [1] * len(target_input_ids)
            token_type_ids += [1] * len(target_input_ids)
            
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        
        return batch_token_ids, batch_mask_ids, batch_token_type_ids
    
    @staticmethod
    def collate_unilm_rl(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_target_ids, batch_schema_ids = [], [], [], [], []
        batch_input_ids = []
        for item in examples:
            (text, instruction_input_ids, instruction_token_type_ids, instruction_attention_mask,
               input_ids, token_type_ids, attention_mask,
               schema_input_ids, schema_token_type_ids, schema_attention_mask,
               target_input_ids, target_token_type_ids, target_attention_mask) = item
            
            # cls instruction sep input sep schema sep target end
            query_input_ids = instruction_input_ids + input_ids + schema_input_ids[:-1]
            end_id = schema_input_ids[-1]
            attention_mask = [1] * len(query_input_ids)
            token_type_ids = [0] * len(query_input_ids)
            
            # batch_schema_ids.append(schema_input_ids)
            batch_token_ids.append(query_input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            batch_target_ids.append(target_input_ids)
            # batch_input_ids.append(instruction_input_ids + input_ids)
        
        # batch_schema_ids = torch.tensor(sequence_padding(batch_schema_ids)).long()
        # batch_input_ids = torch.tensor(sequence_padding(batch_input_ids)).long()
        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).long()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_target_ids = torch.tensor(sequence_padding(batch_target_ids)).long()
        
        batch_size = batch_token_ids.shape[0]
        batch_token_ids = torch.cat([batch_token_ids, end_id*torch.ones((batch_size,1)).long()], dim=1)
        batch_mask_ids = torch.cat([batch_mask_ids, torch.ones((batch_size,1)).long()], dim=1)
        batch_token_type_ids = torch.cat([batch_token_type_ids, torch.zeros((batch_size,1)).long()], dim=1)
        
        return batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_target_ids
    
    @staticmethod
    def collate_t5_v1(examples):
        batch_encoder_token_ids, batch_encoder_mask_ids, batch_encoder_token_type_ids = [], [], []
        batch_decoder_token_ids, batch_decoder_mask_ids = [], []
        for item in examples:
            (text, instruction_input_ids, instruction_token_type_ids, instruction_attention_mask,
               input_ids, token_type_ids, attention_mask,
               schema_input_ids, schema_token_type_ids, schema_attention_mask,
               target_input_ids, target_token_type_ids, target_attention_mask) = item
            
            # instruction sep input sep schema sep target end
            encoder_input_ids = instruction_input_ids + input_ids + schema_input_ids
            encdoer_attention_mask = [1] * len(encoder_input_ids)
            encoder_token_type_ids = [0] * len(encoder_input_ids)
            
            decoder_input_ids = target_input_ids
            decoder_attention_mask = [1] * len(decoder_input_ids)
            
            batch_encoder_token_ids.append(encoder_input_ids)
            batch_encoder_mask_ids.append(encdoer_attention_mask)
            batch_encoder_token_type_ids.append(encoder_token_type_ids)
            
            batch_decoder_token_ids.append(decoder_input_ids)
            batch_decoder_mask_ids.append(decoder_attention_mask)

        batch_encoder_token_ids = torch.tensor(sequence_padding(batch_encoder_token_ids)).long()
        batch_encoder_mask_ids = torch.tensor(sequence_padding(batch_encoder_mask_ids)).float()
        batch_encoder_token_type_ids = torch.tensor(sequence_padding(batch_encoder_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_decoder_input_ids = torch.tensor(sequence_padding(batch_decoder_token_ids)).long()#RoBERTa 不需要NSP
        batch_decoder_mask_ids = torch.tensor(sequence_padding(batch_decoder_mask_ids)).float()#RoBERTa 不需要NSP
        
        return (batch_encoder_token_ids, batch_encoder_mask_ids, batch_encoder_token_type_ids, 
                batch_decoder_input_ids, batch_decoder_mask_ids)
    
    @staticmethod
    def collate_t5_v2(examples):
        batch_encoder_token_ids, batch_encoder_mask_ids, batch_encoder_token_type_ids = [], [], []
        batch_decoder_token_ids, batch_decoder_mask_ids = [], []
        for item in examples:
            (text, instruction_input_ids, instruction_token_type_ids, instruction_attention_mask,
               input_ids, token_type_ids, attention_mask,
               schema_input_ids, schema_token_type_ids, schema_attention_mask,
               target_input_ids, target_token_type_ids, target_attention_mask) = item
            
            # instruction sep input sep schema sep target end
            encoder_input_ids = instruction_input_ids + input_ids
            encdoer_attention_mask = [1] * len(encoder_input_ids)
            encoder_token_type_ids = [0] * len(encoder_input_ids)
            
            decoder_input_ids = schema_input_ids + target_input_ids
            decoder_attention_mask = [1] * len(decoder_input_ids)
            
            batch_encoder_token_ids.append(encoder_input_ids)
            batch_encoder_mask_ids.append(encdoer_attention_mask)
            batch_encoder_token_type_ids.append(encoder_token_type_ids)
            
            batch_decoder_token_ids.append(decoder_input_ids)
            batch_decoder_mask_ids.append(decoder_attention_mask)

        batch_encoder_token_ids = torch.tensor(sequence_padding(batch_encoder_token_ids)).long()
        batch_encoder_mask_ids = torch.tensor(sequence_padding(batch_encoder_mask_ids)).float()
        batch_encoder_token_type_ids = torch.tensor(sequence_padding(batch_encoder_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_decoder_input_ids = torch.tensor(sequence_padding(batch_decoder_token_ids)).long()#RoBERTa 不需要NSP
        batch_decoder_mask_ids = torch.tensor(sequence_padding(batch_decoder_mask_ids)).float()#RoBERTa 不需要NSP
        
        return (batch_encoder_token_ids, batch_encoder_mask_ids, batch_encoder_token_type_ids, 
                batch_decoder_input_ids, batch_decoder_mask_ids)
    
class data_generator_flatten_schema_ccks_casual(data_generator_flatten_schema):
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
                # print(total_target_dict[target_type], '=========before========')
                if self.remove_dup:
                    total_target_dict[target_type] = deleteDuplicate_v1(total_target_dict[target_type])
                after_num = len(total_target_dict[target_type])
                # if before_num != after_num:
                #     print(total_target_dict[target_type], '=========after========')
            
        if flag:
            print(total_target_dict, '========after=========')
            
        span_start = item['span_start']
        span_end = item['span_end']

        span_input_ids = input_ids[span_start:span_end] + self.tokenizer(self.sep_token, add_special_tokens=False)['input_ids']
        span_type_ids = [0] * len(span_input_ids)
        span_attention_mask = [1] * len(span_input_ids)
        
        schema_dict = self.schema_dict
        
        schema_strings = ''
        key_list = list(schema_dict['role2sentinel'])
        if self.schema_shuffle:
            import random
            if random.random() > 0.5:
                random.shuffle(key_list)
                
        for cond in item.get('cond', []):
            for key in cond:
                schema_strings += key + self.sep_token + cond[key] + self.sep_token

        # if self.task_dict['add_schema_type']:
        #     schema_strings += target_type + self.sep_token

        for role in key_list:
            schema_strings += role + schema_dict['role2sentinel'][role] # role sentinel

        schema_strings += self.sep_token
        if self.task_dict.get('add_cardinality', False):
            schema_strings += str(len(total_target_dict[target_type])) + self.sep_token

        encoder_schema_text = self.tokenizer(schema_strings, return_offsets_mapping=True, add_special_tokens=False)
        schema_input_ids = encoder_schema_text["input_ids"]
        schema_token_type_ids = encoder_schema_text.get("token_type_ids", []) #RoBERTa不需要NSP任务
        schema_attention_mask = encoder_schema_text["attention_mask"]
        
        output_list = []
        
        target_strings = ''
        for target_type in total_target_dict:
            key_list = list(schema_dict['role2sentinel'])
            if self.schema_shuffle:
                import random
                if random.random() > 0.5:
                    random.shuffle(key_list)

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
                    if argument:
                        if argument_start_index != -1:
                            start, end = argument_start_index, argument_start_index + len(role_dict['argument']) - 1
                            start_t, end_t = char_span_to_token_span(char2token, (start, end+1))
                            argument_ids = self.tokenizer.encode(argument, add_special_tokens=False)
                            if input_ids[start_t:end_t] and argument_ids and start_t >= span_start and end_t <= span_end:
                                if (argument, schema_dict['role2sentinel'][role_type]) not in target_dict:
                                    target_dict[(argument, schema_dict['role2sentinel'][role_type])] = ''
                            else:
                                if self.greedy_search:
                                    argument_ids = self.tokenizer.encode(argument, add_special_tokens=False)
                                    if self.search_mode == 'token_id':
                                        argument_ids = self.tokenizer.encode(argument, add_special_tokens=False)
                                        sh = search(argument_ids, span_input_ids)
                                    elif self.search_mode == 'string':
                                        span_text_pos = token_span_to_char_span(token2char, [span_start, span_end])
                                        span_text = text[span_text_pos[0]:span_text_pos[1]]
                                        sh = search(argument, span_text)
                                    if sh != -1 and argument_ids:
                                        # target_strings += argument + schema_dict['role2sentinel'][role_type]
                                        if (argument, schema_dict['role2sentinel'][role_type]) not in target_dict:
                                            target_dict[(argument, schema_dict['role2sentinel'][role_type])] = ''

                        else:
                            argument_ids = self.tokenizer.encode(argument, add_special_tokens=False)
                            if self.search_mode == 'token_id':
                                argument_ids = self.tokenizer.encode(argument, add_special_tokens=False)
                                sh = search(argument_ids, span_input_ids)
                            elif self.search_mode == 'string':
                                span_text_pos = token_span_to_char_span(token2char, [span_start, span_end])
                                span_text = text[span_text_pos[0]:span_text_pos[1]]
                                sh = search(argument, span_text)
                            if sh != -1 and argument_ids:
                                # target_strings += argument + schema_dict['role2sentinel'][role_type]
                                if (argument, schema_dict['role2sentinel'][role_type]) not in target_dict:
                                    target_dict[(argument, schema_dict['role2sentinel'][role_type])] = ''

                add_group_flag = False
                for key_set in target_dict:
                    target_strings += "".join(list(key_set))
                    add_group_flag = True
                if self.task_dict['add_schema_type'] and add_group_flag:
                    target_strings += self.seg_token + schema_dict['role2sentinel'][target_type] # add target-type
                if add_group_flag:
                    target_strings += self.group_token
                if flag:
                    print(target_dict, '=====target_dict====')
                    
        target_strings += self.end_token
        encoder_target_text = self.tokenizer(target_strings, return_offsets_mapping=True, max_length=self.max_len, truncation=False, add_special_tokens=False)
        target_input_ids = encoder_target_text["input_ids"]
        target_token_type_ids = encoder_target_text.get("token_type_ids", []) #RoBERTa不需要NSP任务
        target_attention_mask = encoder_target_text["attention_mask"]

        output_list.append((text, instruction_input_ids, instruction_token_type_ids, instruction_attention_mask,
           span_input_ids, span_type_ids, span_attention_mask,
           schema_input_ids, schema_token_type_ids, schema_attention_mask,
           target_input_ids, target_token_type_ids, target_attention_mask))
            
        return output_list
    
    @staticmethod
    def collate_unilm(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids = [], [], []
        for item in examples:
            (text, instruction_input_ids, instruction_token_type_ids, instruction_attention_mask,
               input_ids, token_type_ids, attention_mask,
               schema_input_ids, schema_token_type_ids, schema_attention_mask,
               target_input_ids, target_token_type_ids, target_attention_mask) = item
            
            # cls instruction sep input sep schema sep target end
            input_ids = instruction_input_ids + input_ids + schema_input_ids
            attention_mask = [1] * len(input_ids)
            token_type_ids = [0] * len(input_ids)
            
            input_ids += target_input_ids
            attention_mask += [1] * len(target_input_ids)
            token_type_ids += [1] * len(target_input_ids)
            
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        
        return batch_token_ids, batch_mask_ids, batch_token_type_ids
    
    @staticmethod
    def collate_unilm_rl(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_target_ids, batch_schema_ids = [], [], [], [], []
        batch_input_ids = []
        for item in examples:
            (text, instruction_input_ids, instruction_token_type_ids, instruction_attention_mask,
               input_ids, token_type_ids, attention_mask,
               schema_input_ids, schema_token_type_ids, schema_attention_mask,
               target_input_ids, target_token_type_ids, target_attention_mask) = item
            
            # cls instruction sep input sep schema sep target end
            query_input_ids = instruction_input_ids + input_ids + schema_input_ids[:-1]
            end_id = schema_input_ids[-1]
            attention_mask = [1] * len(query_input_ids)
            token_type_ids = [0] * len(query_input_ids)
            
            # batch_schema_ids.append(schema_input_ids)
            batch_token_ids.append(query_input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            batch_target_ids.append(target_input_ids)
            # batch_input_ids.append(instruction_input_ids + input_ids)
        
        # batch_schema_ids = torch.tensor(sequence_padding(batch_schema_ids)).long()
        # batch_input_ids = torch.tensor(sequence_padding(batch_input_ids)).long()
        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).long()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_target_ids = torch.tensor(sequence_padding(batch_target_ids)).long()
        
        batch_size = batch_token_ids.shape[0]
        batch_token_ids = torch.cat([batch_token_ids, end_id*torch.ones((batch_size,1)).long()], dim=1)
        batch_mask_ids = torch.cat([batch_mask_ids, torch.ones((batch_size,1)).long()], dim=1)
        batch_token_type_ids = torch.cat([batch_token_type_ids, torch.zeros((batch_size,1)).long()], dim=1)
        
        return batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_target_ids
    
    @staticmethod
    def collate_t5_v1(examples):
        batch_encoder_token_ids, batch_encoder_mask_ids, batch_encoder_token_type_ids = [], [], []
        batch_decoder_token_ids, batch_decoder_mask_ids = [], []
        for item in examples:
            (text, instruction_input_ids, instruction_token_type_ids, instruction_attention_mask,
               input_ids, token_type_ids, attention_mask,
               schema_input_ids, schema_token_type_ids, schema_attention_mask,
               target_input_ids, target_token_type_ids, target_attention_mask) = item
            
            # instruction sep input sep schema sep target end
            encoder_input_ids = instruction_input_ids + input_ids + schema_input_ids
            encdoer_attention_mask = [1] * len(encoder_input_ids)
            encoder_token_type_ids = [0] * len(encoder_input_ids)
            
            decoder_input_ids = target_input_ids
            decoder_attention_mask = [1] * len(decoder_input_ids)
            
            batch_encoder_token_ids.append(encoder_input_ids)
            batch_encoder_mask_ids.append(encdoer_attention_mask)
            batch_encoder_token_type_ids.append(encoder_token_type_ids)
            
            batch_decoder_token_ids.append(decoder_input_ids)
            batch_decoder_mask_ids.append(decoder_attention_mask)

        batch_encoder_token_ids = torch.tensor(sequence_padding(batch_encoder_token_ids)).long()
        batch_encoder_mask_ids = torch.tensor(sequence_padding(batch_encoder_mask_ids)).float()
        batch_encoder_token_type_ids = torch.tensor(sequence_padding(batch_encoder_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_decoder_input_ids = torch.tensor(sequence_padding(batch_decoder_token_ids)).long()#RoBERTa 不需要NSP
        batch_decoder_mask_ids = torch.tensor(sequence_padding(batch_decoder_mask_ids)).float()#RoBERTa 不需要NSP
        
        return (batch_encoder_token_ids, batch_encoder_mask_ids, batch_encoder_token_type_ids, 
                batch_decoder_input_ids, batch_decoder_mask_ids)
    
    @staticmethod
    def collate_t5_v2(examples):
        batch_encoder_token_ids, batch_encoder_mask_ids, batch_encoder_token_type_ids = [], [], []
        batch_decoder_token_ids, batch_decoder_mask_ids = [], []
        for item in examples:
            (text, instruction_input_ids, instruction_token_type_ids, instruction_attention_mask,
               input_ids, token_type_ids, attention_mask,
               schema_input_ids, schema_token_type_ids, schema_attention_mask,
               target_input_ids, target_token_type_ids, target_attention_mask) = item
            
            # instruction sep input sep schema sep target end
            encoder_input_ids = instruction_input_ids + input_ids
            encdoer_attention_mask = [1] * len(encoder_input_ids)
            encoder_token_type_ids = [0] * len(encoder_input_ids)
            
            decoder_input_ids = schema_input_ids + target_input_ids
            decoder_attention_mask = [1] * len(decoder_input_ids)
            
            batch_encoder_token_ids.append(encoder_input_ids)
            batch_encoder_mask_ids.append(encdoer_attention_mask)
            batch_encoder_token_type_ids.append(encoder_token_type_ids)
            
            batch_decoder_token_ids.append(decoder_input_ids)
            batch_decoder_mask_ids.append(decoder_attention_mask)

        batch_encoder_token_ids = torch.tensor(sequence_padding(batch_encoder_token_ids)).long()
        batch_encoder_mask_ids = torch.tensor(sequence_padding(batch_encoder_mask_ids)).float()
        batch_encoder_token_type_ids = torch.tensor(sequence_padding(batch_encoder_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_decoder_input_ids = torch.tensor(sequence_padding(batch_decoder_token_ids)).long()#RoBERTa 不需要NSP
        batch_decoder_mask_ids = torch.tensor(sequence_padding(batch_decoder_mask_ids)).float()#RoBERTa 不需要NSP
        
        return (batch_encoder_token_ids, batch_encoder_mask_ids, batch_encoder_token_type_ids, 
                batch_decoder_input_ids, batch_decoder_mask_ids)
    
class data_generator_single_schema_chip_casual(data_generator_single_schema):
    def __init__(self, data, tokenizer, max_len, schema, label=0, 
                 task_dict={}, mode='train', build_data=True, add_neg=False, add_role_shuffle=False,
                doc_stride=32, offset=0):
        
        super().__init__(data, tokenizer, max_len, schema, label, 
                 task_dict, mode, build_data, add_neg, add_role_shuffle,
                doc_stride, offset)
        
    def encoder(self, item):
        text = item["text"]
        
        import random
        if 'candidate_type' in item:
            random.shuffle(item['candidate_type'])
            event_dict = {
                            'type':item['candidate_type'][0],
                            'role_list':[]
                        }
            item['target_list'] = [event_dict]
        
        flag = False
        if '大连百傲化学股份有限公司关于股东股权解质押及再质押的公告' in text:
            print(item, '==========')
            flag = True
        
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
                # print(total_target_dict[target_type], '=========before========')
                if self.remove_dup:
                    total_target_dict[target_type] = deleteDuplicate_v1(total_target_dict[target_type])
                after_num = len(total_target_dict[target_type])
                # if before_num != after_num:
                #     print(total_target_dict[target_type], '=========after========')
            
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
                    
            for cond in item.get('cond', []):
                for key in cond:
                    schema_strings += key + self.sep_token + cond[key] + self.sep_token

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
                    if not argument:
                        continue
                    if argument_start_index != -1:
                        start, end = argument_start_index, argument_start_index + len(role_dict['argument']) - 1
                        start_t, end_t = char_span_to_token_span(char2token, (start, end+1))
                        argument_ids = self.tokenizer.encode(argument, add_special_tokens=False)
                        if input_ids[start_t:end_t] and argument_ids and start_t >= span_start and end_t <= span_end:
                            if (argument, schema_dict['role2sentinel'][role_type]) not in target_dict:
                                target_dict[(argument, schema_dict['role2sentinel'][role_type])] = ''
                        else:
                            if self.greedy_search:
                                argument_ids = self.tokenizer.encode(argument, add_special_tokens=False)
                                if self.search_mode == 'token_id':
                                    argument_ids = self.tokenizer.encode(argument, add_special_tokens=False)
                                    sh = search(argument_ids, span_input_ids)
                                elif self.search_mode == 'string':
                                    span_text_pos = token_span_to_char_span(token2char, [span_start, span_end])
                                    span_text = text[span_text_pos[0]:span_text_pos[1]]
                                    sh = search(argument, span_text)
                                if sh != -1 and argument_ids:
                                    # target_strings += argument + schema_dict['role2sentinel'][role_type]
                                    if (argument, schema_dict['role2sentinel'][role_type]) not in target_dict:
                                        target_dict[(argument, schema_dict['role2sentinel'][role_type])] = ''
                                    
                    else:
                        argument_ids = self.tokenizer.encode(argument, add_special_tokens=False)
                        if self.search_mode == 'token_id':
                            argument_ids = self.tokenizer.encode(argument, add_special_tokens=False)
                            sh = search(argument_ids, span_input_ids)
                        elif self.search_mode == 'string':
                            span_text_pos = token_span_to_char_span(token2char, [span_start, span_end])
                            span_text = text[span_text_pos[0]:span_text_pos[1]]
                            sh = search(argument, span_text)
                        if sh != -1 and argument_ids:
                            # target_strings += argument + schema_dict['role2sentinel'][role_type]
                            if (argument, schema_dict['role2sentinel'][role_type]) not in target_dict:
                                target_dict[(argument, schema_dict['role2sentinel'][role_type])] = ''

                add_group_flag = False
                for key_set in target_dict:
                    target_strings += "".join(list(key_set))
                    add_group_flag = True
                if flag:
                    print(target_dict, '=====target_dict====', target_strings)

                if add_group_flag:
                    target_strings += self.group_token
                    
                if flag:
                    print(target_dict, '=====target_dict====', target_strings)
                    
            # print(target_strings, '==target_strings==')

            target_strings += self.end_token
            encoder_target_text = self.tokenizer(target_strings, return_offsets_mapping=True, max_length=self.max_len, truncation=False, add_special_tokens=False)
            target_input_ids = encoder_target_text["input_ids"]
            target_token_type_ids = encoder_target_text.get("token_type_ids", []) #RoBERTa不需要NSP任务
            target_attention_mask = encoder_target_text["attention_mask"]
            
            if self.task_dict.get('augment_marks', False):
                import random
                if random.random() > 0.4:
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
    
    def __getitem__(self, idx):
        return self.encoder(self.features[idx])[0]
    
    def get_labels(self):
        return self.labels
    
    def get_task_id(self):
        return self._task_id

    @staticmethod
    def collate_unilm(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids = [], [], []
        for item in examples:
            (text, instruction_input_ids, instruction_token_type_ids, instruction_attention_mask,
               input_ids, token_type_ids, attention_mask,
               schema_input_ids, schema_token_type_ids, schema_attention_mask,
               target_input_ids, target_token_type_ids, target_attention_mask) = item
            
            # cls instruction sep input sep schema sep target end
            input_ids = instruction_input_ids + input_ids + schema_input_ids
            attention_mask = [1] * len(input_ids)
            token_type_ids = [0] * len(input_ids)
            
            input_ids += target_input_ids
            attention_mask += [1] * len(target_input_ids)
            token_type_ids += [1] * len(target_input_ids)
            
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        
        return batch_token_ids, batch_mask_ids, batch_token_type_ids
    
    @staticmethod
    def collate_unilm_rl(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_target_ids, batch_schema_ids = [], [], [], [], []
        batch_input_ids = []
        for item in examples:
            (text, instruction_input_ids, instruction_token_type_ids, instruction_attention_mask,
               input_ids, token_type_ids, attention_mask,
               schema_input_ids, schema_token_type_ids, schema_attention_mask,
               target_input_ids, target_token_type_ids, target_attention_mask) = item
            
            # cls instruction sep input sep schema sep target end
            query_input_ids = instruction_input_ids + input_ids + schema_input_ids[:-1]
            end_id = schema_input_ids[-1]
            attention_mask = [1] * len(query_input_ids)
            token_type_ids = [0] * len(query_input_ids)
            
            # batch_schema_ids.append(schema_input_ids)
            batch_token_ids.append(query_input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            batch_target_ids.append(target_input_ids)
            # batch_input_ids.append(instruction_input_ids + input_ids)
        
        # batch_schema_ids = torch.tensor(sequence_padding(batch_schema_ids)).long()
        # batch_input_ids = torch.tensor(sequence_padding(batch_input_ids)).long()
        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).long()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_target_ids = torch.tensor(sequence_padding(batch_target_ids)).long()
        
        batch_size = batch_token_ids.shape[0]
        batch_token_ids = torch.cat([batch_token_ids, end_id*torch.ones((batch_size,1)).long()], dim=1)
        batch_mask_ids = torch.cat([batch_mask_ids, torch.ones((batch_size,1)).long()], dim=1)
        batch_token_type_ids = torch.cat([batch_token_type_ids, torch.zeros((batch_size,1)).long()], dim=1)
        
        return batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_target_ids
    
    @staticmethod
    def collate_t5_v1(examples):
        batch_encoder_token_ids, batch_encoder_mask_ids, batch_encoder_token_type_ids = [], [], []
        batch_decoder_token_ids, batch_decoder_mask_ids = [], []
        for item in examples:
            (text, instruction_input_ids, instruction_token_type_ids, instruction_attention_mask,
               input_ids, token_type_ids, attention_mask,
               schema_input_ids, schema_token_type_ids, schema_attention_mask,
               target_input_ids, target_token_type_ids, target_attention_mask) = item
            
            # instruction sep input sep schema sep target end
            encoder_input_ids = instruction_input_ids + input_ids + schema_input_ids
            encdoer_attention_mask = [1] * len(encoder_input_ids)
            encoder_token_type_ids = [0] * len(encoder_input_ids)
            
            decoder_input_ids = target_input_ids
            decoder_attention_mask = [1] * len(decoder_input_ids)
            
            batch_encoder_token_ids.append(encoder_input_ids)
            batch_encoder_mask_ids.append(encdoer_attention_mask)
            batch_encoder_token_type_ids.append(encoder_token_type_ids)
            
            batch_decoder_token_ids.append(decoder_input_ids)
            batch_decoder_mask_ids.append(decoder_attention_mask)

        batch_encoder_token_ids = torch.tensor(sequence_padding(batch_encoder_token_ids)).long()
        batch_encoder_mask_ids = torch.tensor(sequence_padding(batch_encoder_mask_ids)).float()
        batch_encoder_token_type_ids = torch.tensor(sequence_padding(batch_encoder_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_decoder_input_ids = torch.tensor(sequence_padding(batch_decoder_token_ids)).long()#RoBERTa 不需要NSP
        batch_decoder_mask_ids = torch.tensor(sequence_padding(batch_decoder_mask_ids)).float()#RoBERTa 不需要NSP
        
        return (batch_encoder_token_ids, batch_encoder_mask_ids, batch_encoder_token_type_ids, 
                batch_decoder_input_ids, batch_decoder_mask_ids)
    
    @staticmethod
    def collate_t5_v2(examples):
        batch_encoder_token_ids, batch_encoder_mask_ids, batch_encoder_token_type_ids = [], [], []
        batch_decoder_token_ids, batch_decoder_mask_ids = [], []
        for item in examples:
            (text, instruction_input_ids, instruction_token_type_ids, instruction_attention_mask,
               input_ids, token_type_ids, attention_mask,
               schema_input_ids, schema_token_type_ids, schema_attention_mask,
               target_input_ids, target_token_type_ids, target_attention_mask) = item
            
            # instruction sep input sep schema sep target end
            encoder_input_ids = instruction_input_ids + input_ids
            encdoer_attention_mask = [1] * len(encoder_input_ids)
            encoder_token_type_ids = [0] * len(encoder_input_ids)
            
            decoder_input_ids = schema_input_ids + target_input_ids
            decoder_attention_mask = [1] * len(decoder_input_ids)
            
            batch_encoder_token_ids.append(encoder_input_ids)
            batch_encoder_mask_ids.append(encdoer_attention_mask)
            batch_encoder_token_type_ids.append(encoder_token_type_ids)
            
            batch_decoder_token_ids.append(decoder_input_ids)
            batch_decoder_mask_ids.append(decoder_attention_mask)

        batch_encoder_token_ids = torch.tensor(sequence_padding(batch_encoder_token_ids)).long()
        batch_encoder_mask_ids = torch.tensor(sequence_padding(batch_encoder_mask_ids)).float()
        batch_encoder_token_type_ids = torch.tensor(sequence_padding(batch_encoder_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_decoder_input_ids = torch.tensor(sequence_padding(batch_decoder_token_ids)).long()#RoBERTa 不需要NSP
        batch_decoder_mask_ids = torch.tensor(sequence_padding(batch_decoder_mask_ids)).float()#RoBERTa 不需要NSP
        
        return (batch_encoder_token_ids, batch_encoder_mask_ids, batch_encoder_token_type_ids, 
                batch_decoder_input_ids, batch_decoder_mask_ids)
    
    
class data_generator_flatten_schema_chip_casual(data_generator_flatten_schema):
    def __init__(self, data, tokenizer, max_len, schema, label=0, 
                 task_dict={}, mode='train', build_data=True, add_neg=False, add_role_shuffle=False,
                doc_stride=32, offset=0):
        
        super().__init__(data, tokenizer, max_len, schema, label, 
                 task_dict, mode, build_data, add_neg, add_role_shuffle,
                doc_stride, offset)
        
    def encoder(self, item):
        text = item["text"]
        
        flag = False
        if '大连百傲化学股份有限公司关于股东股权解质押及再质押的公告' in text:
            print(item, '==========')
            flag = True
        
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
                # print(total_target_dict[target_type], '=========before========')
                if self.remove_dup:
                    total_target_dict[target_type] = deleteDuplicate_v1(total_target_dict[target_type])
                after_num = len(total_target_dict[target_type])
                # if before_num != after_num:
                #     print(total_target_dict[target_type], '=========after========')
            
        if flag:
            print(total_target_dict, '========after=========')
            
        span_start = item['span_start']
        span_end = item['span_end']

        span_input_ids = input_ids[span_start:span_end] + self.tokenizer(self.sep_token, add_special_tokens=False)['input_ids']
        span_type_ids = [0] * len(span_input_ids)
        span_attention_mask = [1] * len(span_input_ids)
        
        schema_dict = self.schema_dict
        
        schema_strings = ''
        key_list = list(schema_dict['role2sentinel'])
        if self.schema_shuffle:
            import random
            if random.random() > 0.5:
                random.shuffle(key_list)
                
        for cond in item.get('cond', []):
            for key in cond:
                schema_strings += key + self.sep_token + cond[key] + self.sep_token

        for role in key_list:
            schema_strings += role + schema_dict['role2sentinel'][role] # role sentinel

        schema_strings += self.sep_token
        encoder_schema_text = self.tokenizer(schema_strings, return_offsets_mapping=True, max_length=self.max_len, truncation=False, add_special_tokens=False)
        schema_input_ids = encoder_schema_text["input_ids"]
        schema_token_type_ids = encoder_schema_text.get("token_type_ids", []) #RoBERTa不需要NSP任务
        schema_attention_mask = encoder_schema_text["attention_mask"]
        
        output_list = []
        target_strings = ''
        
        for target_type in total_target_dict:
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
                    if not argument:
                        continue
                    if argument_start_index != -1:
                        start, end = argument_start_index, argument_start_index + len(role_dict['argument']) - 1
                        start_t, end_t = char_span_to_token_span(char2token, (start, end+1))
                        argument_ids = self.tokenizer.encode(argument, add_special_tokens=False)
                        if input_ids[start_t:end_t] and argument_ids and start_t >= span_start and end_t <= span_end:
                            if (argument, schema_dict['role2sentinel'][role_type]) not in target_dict:
                                target_dict[(argument, schema_dict['role2sentinel'][role_type])] = ''
                        else:
                            if self.greedy_search:
                                argument_ids = self.tokenizer.encode(argument, add_special_tokens=False)
                                if self.search_mode == 'token_id':
                                    argument_ids = self.tokenizer.encode(argument, add_special_tokens=False)
                                    sh = search(argument_ids, span_input_ids)
                                elif self.search_mode == 'string':
                                    span_text_pos = token_span_to_char_span(token2char, [span_start, span_end])
                                    span_text = text[span_text_pos[0]:span_text_pos[1]]
                                    sh = search(argument, span_text)
                                if sh != -1 and argument_ids:
                                    # target_strings += argument + schema_dict['role2sentinel'][role_type]
                                    if (argument, schema_dict['role2sentinel'][role_type]) not in target_dict:
                                        target_dict[(argument, schema_dict['role2sentinel'][role_type])] = ''
                                    
                    else:
                        argument_ids = self.tokenizer.encode(argument, add_special_tokens=False)
                        if self.search_mode == 'token_id':
                            argument_ids = self.tokenizer.encode(argument, add_special_tokens=False)
                            sh = search(argument_ids, span_input_ids)
                        elif self.search_mode == 'string':
                            span_text_pos = token_span_to_char_span(token2char, [span_start, span_end])
                            span_text = text[span_text_pos[0]:span_text_pos[1]]
                            sh = search(argument, span_text)
                        if sh != -1 and argument_ids:
                            # target_strings += argument + schema_dict['role2sentinel'][role_type]
                            if (argument, schema_dict['role2sentinel'][role_type]) not in target_dict:
                                target_dict[(argument, schema_dict['role2sentinel'][role_type])] = ''

                add_group_flag = False
                for key_set in target_dict:
                    target_strings += "".join(list(key_set))
                    add_group_flag = True
                if self.task_dict['add_schema_type'] and add_group_flag:
                    target_strings += self.seg_token + schema_dict['role2sentinel'][target_type] # add target-type
                if add_group_flag:
                    target_strings += self.group_token
                if flag:
                    print(target_dict, '=====target_dict====')

        target_strings += self.end_token
        encoder_target_text = self.tokenizer(target_strings, return_offsets_mapping=True, max_length=self.max_len, truncation=False, add_special_tokens=False)
        target_input_ids = encoder_target_text["input_ids"]
        target_token_type_ids = encoder_target_text.get("token_type_ids", []) #RoBERTa不需要NSP任务
        target_attention_mask = encoder_target_text["attention_mask"]

        if self.task_dict.get('augment_marks', False):
            import random
            if random.random() > 0.4:
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
    
    def __getitem__(self, idx):
        return self.encoder(self.features[idx])[0]
    
    def get_labels(self):
        return self.labels
    
    def get_task_id(self):
        return self._task_id

    @staticmethod
    def collate_unilm(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids = [], [], []
        for item in examples:
            (text, instruction_input_ids, instruction_token_type_ids, instruction_attention_mask,
               input_ids, token_type_ids, attention_mask,
               schema_input_ids, schema_token_type_ids, schema_attention_mask,
               target_input_ids, target_token_type_ids, target_attention_mask) = item
            
            # cls instruction sep input sep schema sep target end
            input_ids = instruction_input_ids + input_ids + schema_input_ids
            attention_mask = [1] * len(input_ids)
            token_type_ids = [0] * len(input_ids)
            
            input_ids += target_input_ids
            attention_mask += [1] * len(target_input_ids)
            token_type_ids += [1] * len(target_input_ids)
            
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        
        return batch_token_ids, batch_mask_ids, batch_token_type_ids
    
    @staticmethod
    def collate_unilm_rl(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_target_ids, batch_schema_ids = [], [], [], [], []
        batch_input_ids = []
        for item in examples:
            (text, instruction_input_ids, instruction_token_type_ids, instruction_attention_mask,
               input_ids, token_type_ids, attention_mask,
               schema_input_ids, schema_token_type_ids, schema_attention_mask,
               target_input_ids, target_token_type_ids, target_attention_mask) = item
            
            # cls instruction sep input sep schema sep target end
            query_input_ids = instruction_input_ids + input_ids + schema_input_ids[:-1]
            end_id = schema_input_ids[-1]
            attention_mask = [1] * len(query_input_ids)
            token_type_ids = [0] * len(query_input_ids)
            
            # batch_schema_ids.append(schema_input_ids)
            batch_token_ids.append(query_input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            batch_target_ids.append(target_input_ids)
            # batch_input_ids.append(instruction_input_ids + input_ids)
        
        # batch_schema_ids = torch.tensor(sequence_padding(batch_schema_ids)).long()
        # batch_input_ids = torch.tensor(sequence_padding(batch_input_ids)).long()
        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).long()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_target_ids = torch.tensor(sequence_padding(batch_target_ids)).long()
        
        batch_size = batch_token_ids.shape[0]
        batch_token_ids = torch.cat([batch_token_ids, end_id*torch.ones((batch_size,1)).long()], dim=1)
        batch_mask_ids = torch.cat([batch_mask_ids, torch.ones((batch_size,1)).long()], dim=1)
        batch_token_type_ids = torch.cat([batch_token_type_ids, torch.zeros((batch_size,1)).long()], dim=1)
        
        return batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_target_ids
    
    @staticmethod
    def collate_t5_v1(examples):
        batch_encoder_token_ids, batch_encoder_mask_ids, batch_encoder_token_type_ids = [], [], []
        batch_decoder_token_ids, batch_decoder_mask_ids = [], []
        for item in examples:
            (text, instruction_input_ids, instruction_token_type_ids, instruction_attention_mask,
               input_ids, token_type_ids, attention_mask,
               schema_input_ids, schema_token_type_ids, schema_attention_mask,
               target_input_ids, target_token_type_ids, target_attention_mask) = item
            
            # instruction sep input sep schema sep target end
            encoder_input_ids = instruction_input_ids + input_ids + schema_input_ids
            encdoer_attention_mask = [1] * len(encoder_input_ids)
            encoder_token_type_ids = [0] * len(encoder_input_ids)
            
            decoder_input_ids = target_input_ids
            decoder_attention_mask = [1] * len(decoder_input_ids)
            
            batch_encoder_token_ids.append(encoder_input_ids)
            batch_encoder_mask_ids.append(encdoer_attention_mask)
            batch_encoder_token_type_ids.append(encoder_token_type_ids)
            
            batch_decoder_token_ids.append(decoder_input_ids)
            batch_decoder_mask_ids.append(decoder_attention_mask)

        batch_encoder_token_ids = torch.tensor(sequence_padding(batch_encoder_token_ids)).long()
        batch_encoder_mask_ids = torch.tensor(sequence_padding(batch_encoder_mask_ids)).float()
        batch_encoder_token_type_ids = torch.tensor(sequence_padding(batch_encoder_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_decoder_input_ids = torch.tensor(sequence_padding(batch_decoder_token_ids)).long()#RoBERTa 不需要NSP
        batch_decoder_mask_ids = torch.tensor(sequence_padding(batch_decoder_mask_ids)).float()#RoBERTa 不需要NSP
        
        return (batch_encoder_token_ids, batch_encoder_mask_ids, batch_encoder_token_type_ids, 
                batch_decoder_input_ids, batch_decoder_mask_ids)
    
    @staticmethod
    def collate_t5_v2(examples):
        batch_encoder_token_ids, batch_encoder_mask_ids, batch_encoder_token_type_ids = [], [], []
        batch_decoder_token_ids, batch_decoder_mask_ids = [], []
        for item in examples:
            (text, instruction_input_ids, instruction_token_type_ids, instruction_attention_mask,
               input_ids, token_type_ids, attention_mask,
               schema_input_ids, schema_token_type_ids, schema_attention_mask,
               target_input_ids, target_token_type_ids, target_attention_mask) = item
            
            # instruction sep input sep schema sep target end
            encoder_input_ids = instruction_input_ids + input_ids
            encdoer_attention_mask = [1] * len(encoder_input_ids)
            encoder_token_type_ids = [0] * len(encoder_input_ids)
            
            decoder_input_ids = schema_input_ids + target_input_ids
            decoder_attention_mask = [1] * len(decoder_input_ids)
            
            batch_encoder_token_ids.append(encoder_input_ids)
            batch_encoder_mask_ids.append(encdoer_attention_mask)
            batch_encoder_token_type_ids.append(encoder_token_type_ids)
            
            batch_decoder_token_ids.append(decoder_input_ids)
            batch_decoder_mask_ids.append(decoder_attention_mask)

        batch_encoder_token_ids = torch.tensor(sequence_padding(batch_encoder_token_ids)).long()
        batch_encoder_mask_ids = torch.tensor(sequence_padding(batch_encoder_mask_ids)).float()
        batch_encoder_token_type_ids = torch.tensor(sequence_padding(batch_encoder_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_decoder_input_ids = torch.tensor(sequence_padding(batch_decoder_token_ids)).long()#RoBERTa 不需要NSP
        batch_decoder_mask_ids = torch.tensor(sequence_padding(batch_decoder_mask_ids)).float()#RoBERTa 不需要NSP
        
        return (batch_encoder_token_ids, batch_encoder_mask_ids, batch_encoder_token_type_ids, 
                batch_decoder_input_ids, batch_decoder_mask_ids)