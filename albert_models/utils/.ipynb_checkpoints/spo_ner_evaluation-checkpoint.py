
import torch
import io
import torch.nn.functional as F
import random
import numpy as np
import time
import math
import datetime
import json
import torch.nn as nn
import logging
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
import json
from utils import spo_dataloader

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

class SPO(tuple):
    """用来存三元组的类
    表现跟tuple基本一致，只是重写了 __hash__ 和 __eq__ 方法，
    使得在判断两个三元组是否等价时容错性更好。
    """
    def __init__(self, spo, predicate=None):
        if predicate:
            self.spox = (
                spo[0],
                predicate,
                spo[2]
            )
        else:
            self.spox = (
                spo[0],
                spo[1],
                spo[2]
            )

    def __hash__(self):
        return self.spox.__hash__()

    def __eq__(self, spo):
        return self.spox == spo.spox
    
def sigmoid(z):
    return 1/(1 + np.exp(-z))
    
def evaluate_spo(rank, task_name, net,
                   device, eval_examples, 
                   tokenizer, max_seq_length, output_file, 
                   schema2id, id2schema, entity_threshold_probs,
                head_threshold_probs, tail_threshold_probs):
    
    entity_threshold = np.log(entity_threshold_probs/(1-entity_threshold_probs))
    head_threshold = np.log(head_threshold_probs/(1-head_threshold_probs))
    tail_threshold = np.log(tail_threshold_probs/(1-tail_threshold_probs))
    
    logger.info("entity_threshold=%.5f, head_threshold=%.5f,tail_threshold=%.5f", entity_threshold,
               head_threshold, tail_threshold)
    net.eval()

    predict_results = []
    conformal_results = []
    
    if task_name == "jdner_spo":
        convert_examples_to_features_fn = spo_dataloader.convert_examples_to_features
        
    if rank == 0:
        logger.info(" convert_examples_to_features_fn ")
        logger.info(convert_examples_to_features_fn)
    
    X, Y, Z = 1e-10, 1e-10, 1e-10
    output_dict_list = []
    result = {}
    output_spo_list = []
    
    for index, eval_example in enumerate(eval_examples):
        feature = convert_examples_to_features_fn([eval_example], schema2id,
                                            max_seq_length,
                                            tokenizer
                                            )[0]
        
        input_ids = torch.Tensor(feature.input_ids).type(torch.long).unsqueeze(0).to(device) 
        input_mask = torch.Tensor(feature.input_mask).type(torch.long).unsqueeze(0).to(device) 
        segment_ids = torch.Tensor(feature.segment_ids).type(torch.long).unsqueeze(0).to(device)

        with torch.no_grad():
            # [1, seq, seq]
            [entity_logits, head_logits, tail_logits] = net(input_ids, input_mask, segment_ids)
        
        # [num_classes, seq, seq]
        entity_logits = entity_logits.cpu()[0]
        # [num_classes, seq, seq]
        head_logits = head_logits.cpu()[0]
        # [num_classes, seq, seq]
        tail_logits = tail_logits.cpu()[0]
            
        entity_logits[:, [0, -1]] -= np.inf
        entity_logits[:, :, [0, -1]] -= np.inf
        subjects, objects = set(), set()
        
        token_mapping = feature.token_mappings
        text = eval_example['text']
        
        for l, h, t in zip(*np.where(entity_logits > entity_threshold)):
            if l == 0:
                subjects.add((h, t, sigmoid(entity_logits[l, h, t])))
            else:
                objects.add((h, t, sigmoid(entity_logits[l, h, t])))
                
        spoes = set()
        for sh, st, s_prob in subjects:
            for oh, ot, o_prob in objects:
                p1s = np.where(head_logits[:, sh, oh] > head_threshold)[0]
                p2s = np.where(tail_logits[:, st, ot] > tail_threshold)[0]
                ps = set(p1s) & set(p2s)
                for p in ps:
                    # (subject, predicate, object)
                    ps_prob = sigmoid(head_logits[p, sh, oh])
                    po_prob = sigmoid(tail_logits[p, st, ot])
                    
                    spoes.add((
                        text[token_mapping[sh-1][0]:token_mapping[st-1][-1] + 1], id2schema[p],
                        text[token_mapping[oh-1][0]:token_mapping[ot-1][-1] + 1],
                        sh, st, oh, ot, s_prob, o_prob, ps_prob, po_prob
                    ))
                    
        spo_list = []
        for spo in list(spoes):
            spo_list.append({"predicate":spo[1].split("_")[1], 
                             "object":{"@value":spo[2]}, 
                             "object_type": {"@value": spo[1].split("_")[2]},
                             "subject":spo[0], 
                             "subject_type":spo[1].split("_")[0]
                             })
        output_spo_list.append({
            'text': eval_example['text'],
            'spo_list': spo_list
        })
                            
        tmp_dict = {
            "example": eval_example,
            'feature': feature,
            'spoes': spoes
        }
        output_dict_list.append(tmp_dict)
        
        if eval_example.get('spo_list', None):
            true_spoes = set()
            for spo in eval_example['spo_list']:
                true_spoes.add((
                    spo['subject'],
                    spo['predicate'],
                    spo['object']['@value']
                ))

            R = set([SPO(spo, predicate=spo[1].split('_')[1]) for spo in spoes])
            T = set([SPO(spo) for spo in true_spoes])

            X += len(R & T)
            Y += len(R)
            Z += len(T)
                            
    import os, json
    with open(output_file+".spo", 'w', encoding='utf-8') as f:
        for item in output_spo_list:
            f.write(json.dumps(item, ensure_ascii=False)+"\n")
            
    f1, precision, recall = 2 * X / (Y + Z + 1e-10), X / (Y + 1e-10), X / (Z + 1e-10)
    
    result['spo_f1'] = f1
    result['spo_precision'] = precision
    result['spo_recall'] = recall
    
    logger.info("== spo result ==")
    logger.info(json.dumps(result, ensure_ascii=False))
            
    return output_dict_list, result
    
def transform_spo2ner(output_file, output_dict_list, id2schema):
    
    X, Y, Z = 1e-10, 1e-10, 1e-10
    X_b, Y_b, Z_b = 1e-10, 1e-10, 1e-10
    
    result = {}
    conformal_results = []
    predict_results = []
    
    for output_dict in output_dict_list:
        eval_feature = output_dict['feature']
        spoes = output_dict['spoes']
        eval_example = output_dict['example']
        token_mapping = eval_feature.token_mappings
        text = eval_example['text']

        entities = []
        recall_entities = set()
        recall_entities_boundary = set()
        for spo in spoes:
            predicate_type = spo[1].split("_")[1],
            subject_type = spo[1].split("_")[0]
            object_type = spo[1].split("_")[2]
            sh = spo[3] - 1
            st = spo[4] - 1
            oh = spo[5] - 1
            ot = spo[6] - 1

            s_prob = spo[7]
            o_prob = spo[8]
            ps_prob = spo[9]    
            po_prob = spo[10]

            subject_entitie = {
                    "start_idx": token_mapping[sh][0],
                    "end_idx": token_mapping[st][-1],
                    "entity": text[token_mapping[sh][0]: token_mapping[st][-1]+1],
                    "type": subject_type,
                    "entity_prob": float(s_prob),
                    "type_prob": float(ps_prob)
            }
            object_entitie = {
                    "start_idx": token_mapping[oh][0],
                    "end_idx": token_mapping[ot][-1],
                    "entity": text[token_mapping[oh][0]: token_mapping[ot][-1]+1],
                    "type": object_type,
                    "entity_prob":float(o_prob),
                    "type_prob": float(po_prob)
            }

            entities.append(subject_entitie)
            entities.append(object_entitie)

            recall_entities.add((token_mapping[sh][0], token_mapping[st][-1], subject_type))
            recall_entities.add((token_mapping[oh][0], token_mapping[ot][-1], object_type))
            recall_entities_boundary.add((token_mapping[sh][0], token_mapping[st][-1], 1))
            recall_entities_boundary.add((token_mapping[oh][0], token_mapping[ot][-1], 1))
                            
        if eval_feature.spoes:
            true_entities = set()
            true_entities_boundary = set()
            true_spoes = eval_feature.spoes
            for spo in true_spoes:
                predicate = id2schema[spo[2]]
                predicate_type = predicate.split("_")[1],
                subject_type = predicate.split("_")[0]
                object_type = predicate.split("_")[2]

                sh = spo[0] - 1
                st = spo[1] - 1
                oh = spo[3] - 1
                ot = spo[4] - 1

                true_entities.add((token_mapping[sh][0], token_mapping[st][-1], subject_type))
                true_entities.add((token_mapping[oh][0], token_mapping[ot][-1], object_type))
                true_entities_boundary.add((token_mapping[sh][0], token_mapping[st][-1], 1))
                true_entities_boundary.add((token_mapping[oh][0], token_mapping[ot][-1], 1))

            X += len(recall_entities & true_entities)
            Y += len(recall_entities)
            Z += len(true_entities)

            X_b += len(recall_entities_boundary & true_entities_boundary)
            Y_b += len(recall_entities_boundary)
            Z_b += len(true_entities_boundary)
                            
        label = len(eval_example['text']) * ['O']
        pred_entities = {}
        for entity in entities:
            if (entity['start_idx'], entity['end_idx']) not in pred_entities:
                pred_entities[(entity['start_idx'], entity['end_idx'])] = []
            pred_entities[(entity['start_idx'], entity['end_idx'])].append(entity)                
        
        pred_entities_final = []
        for key in pred_entities:
            entitie_ = sorted(pred_entities[key], 
                              key=lambda item:item['entity_prob']+item['type_prob'], 
                              reverse=True)[0]
            pred_entities_final.append(entitie_)
                            
        label = len(eval_example['text']) * ['O']
        for _preditc in pred_entities_final:
            if 'I' in label[_preditc['start_idx']]:
                continue
            if 'B' in label[_preditc['start_idx']] and 'O' not in label[_preditc['end_idx']]:
                continue
            if 'O' in label[_preditc['start_idx']] and 'B' in label[_preditc['end_idx']]:
                continue

            label[_preditc['start_idx']] = 'B-' +  _preditc['type']
            label[_preditc['start_idx']+1: _preditc['end_idx']+1] = (_preditc['end_idx'] - _preditc['start_idx']) * [('I-' +  _preditc['type'])]
                            
        predict_results.append([eval_example['text'], label])
        conformal_results.append({"text":eval_example['text'], "results":entities})
        
    f1, precision, recall = 2 * X / (Y + Z + 1e-10), X / (Y+1e-10), X / (Z+1e-10)
    f1_boundary, precision_boundary, recall_boundary = 2 * X_b / (Y_b + Z_b + 1e-10), X_b / (Y_b+1e-10), X_b / (Z_b+1e-10)

    result['f1'] = f1
    result['precision'] = precision
    result['recall'] = recall

    result['f1_boundary'] = f1_boundary
    result['precision_boundary'] = precision_boundary
    result['recall_boundary'] = recall_boundary
    
    import os, json
    logger.info("== spo2ner result ==")
    logger.info(json.dumps(result, ensure_ascii=False))
        
    import os, json
    with open(output_file+".ner", 'w', encoding='utf-8') as f:
        for _result in predict_results:
            for word, tag in zip(_result[0], _result[1]):
                f.write(f'{word} {tag}\n')
            f.write('\n')
                            
    import os, json
    with open(output_file+".ner.conformal", 'w', encoding='utf-8') as f:
        for item in conformal_results:
            f.write(json.dumps(item, ensure_ascii=False)+"\n")
    
    return result
    
            
        
    
    
                
       