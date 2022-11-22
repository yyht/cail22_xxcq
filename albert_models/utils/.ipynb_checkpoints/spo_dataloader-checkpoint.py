# Copyright (c) 2019, Facebook, Inc. and its affiliates. All Rights Reserved
"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function

import logging
import os, copy
import time, random

import torch
from torch.utils.data.dataset import Dataset

import sys
from sklearn.model_selection import train_test_split

import argparse
import csv
import logging
import os
import random
import sys, json
import time

import numpy as np
import pandas as pd
import torch
from utils import ner_utils
import copy
import re
import unicodedata
from collections import namedtuple

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

InputExample = namedtuple('InputExample', ['guid', 'text_a', 
                                   'text_b', 'label'])

InputFeatures = namedtuple('InputFeatures', ['input_ids', 'input_mask', 
                                   'segment_ids', 'entity_labels',
                                'head_labels', 'tail_labels',
                                  'token_mappings',
                                   'spoes'])

def convert_examples_to_features(examples, schema2id, max_seq_length,
                                 tokenizer, **args):
    """Loads a data file into a list of `InputBatch`s."""

    max_length = 0

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0 and args.get("show_case", False):
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        
        tokens = tokenizer.tokenize(example['text'])
        try:
            token_mapping = ner_utils.get_token_mapping(example['text'], tokens)
        except:
            logger.info("== invalid token mapping ==")
            invalid_dict = {
                "text": example['text'],
                "tokens": tokens
            }
            logger.info(json.dumps(invalid_dict, ensure_ascii=False))
            continue
        start_mapping = {j[0]: i for i, j in enumerate(token_mapping) if j}
        end_mapping = {j[-1]: i for i, j in enumerate(token_mapping) if j}
        
        spoes = set()
        entity_labels = []
        head_labels = []
        tail_labels = []
        if example.get('spo_list', None):
            add_flag = False
            for spo_dict in example['spo_list']:
                s = spo_dict['subject']
                o = spo_dict['object']
                s_t = spo_dict['subject_type']
                o_t = spo_dict['object_type']

                schema_key = spo_dict["subject_type"]+"_"+spo_dict["predicate"]+"_"+spo_dict["object_type"]['@value']
                p = schema2id[schema_key]

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

            if not spoes:
                continue
            entity_labels = [set() for i in range(2)]
            head_labels = [set() for i in range(len(schema2id))]
            tail_labels = [set() for i in range(len(schema2id))]
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
                    
            if add_flag and not entity_empty and not head_empty and not tail_empty:
                entity_labels = ner_utils.sequence_padding([list(l) for l in entity_labels])
                head_labels = ner_utils.sequence_padding([list(l) for l in head_labels])
                tail_labels = ner_utils.sequence_padding([list(l) for l in tail_labels])
            else:
                logger.info("*** invalid ***")
                logger.info(json.dumps(spoes, ensure_ascii=False))
                continue
                                   
        tokens = ['[CLS]'] + tokens + ["[SEP]"]
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
            
        input_mask = [1]*len(input_ids)
        segment_ids = [0]*len(input_ids)
    
        if ex_index < 5 and args.get("show_case", False):
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_string: %s" % " ".join([str(x) for x in example.text_a]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            
        
        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              entity_labels=entity_labels,
                              head_labels=head_labels,
                              tail_labels=tail_labels,
                              token_mappings=token_mapping,
                              spoes=spoes))
        # if len(features) >= 1024:
        #     break
    if len(features) >= 1024:
        logger.info("*** features ***")
        logger.info(len(features))
    return features