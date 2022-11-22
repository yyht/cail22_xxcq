

import os, sys
from collections import Counter

import argparse
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_folder",
        default=None,
        type=str,
        help="The config file.", )
    
    parser.add_argument(
        "--output_file",
        default=None,
        type=str,
        help="The config file.", )
    
    parser.add_argument(
        "--majority_vote",
        default=3,
        type=int,
        help="The config file.", )

    args = parser.parse_args()

    return args

args = parse_args() # 从命令行获取

import os
cur_dir_path = os.path.dirname(__file__)
sys.path.extend([cur_dir_path])

prediction_path = os.path.join(cur_dir_path, args.input_folder)
data_dict = {}
import json

from functools import reduce
def deleteDuplicate_v1(input_dict_lst):
    f = lambda x,y:x if y in x else x + [y]
    return reduce(f, [[], ] + input_dict_lst)

for sub_path in os.listdir(prediction_path):
    if '.json' not in sub_path:
        continue
    with open(os.path.join(prediction_path, sub_path)) as frobj:
        for line in frobj:
            content = json.loads(line.strip())
            ID = (content['articleId'], content['sentID'])
            if ID not in data_dict:
                data_dict[ID] = {
                    'sentText':content['sentText'],
                    'spo':Counter()
                }
            spo_list = deleteDuplicate_v1(content['relationMentions'])
            for spo in spo_list:
                spo_tuple = ('em1Text', spo['em1Text'], 
                             'em2Text', spo['em2Text'], 
                            spo['label'])
                data_dict[ID]['spo'][spo_tuple] += 1

vote = int(args.majority_vote)

with open(args.output_file, 'w') as fwobj:
    for key in data_dict:
        article_id, sent_id = key
        tmp_dict = {
            'sentID':sent_id,
            'sentText':data_dict[key]['sentText'],
            'relationMentions':[],
            'entityMentions':[],
            'articleId':article_id
        }
        for spo_key in data_dict[key]['spo']:
            if data_dict[key]['spo'][spo_key] >= vote:
                spo_dict = {
                    'em1Text':spo_key[1],
                    'em2Text':spo_key[3],
                    'label':spo_key[-1]
                }
                tmp_dict['relationMentions'].append(spo_dict)
        fwobj.write(json.dumps(tmp_dict, ensure_ascii=False)+'\n')