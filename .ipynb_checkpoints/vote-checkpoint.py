import os
from collections import Counter

import os, sys
from collections import Counter

"""
vote
"""

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

from functools import reduce
import json
def deleteDuplicate_v1(input_dict_lst):
    f = lambda x,y:x if y in x else x + [y]
    return reduce(f, [[], ] + input_dict_lst)


valid_sub_path = [
    'unilm_roformer_large_v5.json',
    'ie_baffine_v1.json',
    'ie_baffine_v2.json',
    'roformer_negative_learing.json'
]


# wr.write(json.dumps(
#     {"articleId": "666", "sentId": "6", "entityMentions": [], "sentText": text, "relationMentions": spo_list},
#     ensure_ascii=False))
data_cnt = 0
for sub_path in os.listdir(prediction_path):
    if sub_path not in valid_sub_path:
        continue
    print(os.path.join(prediction_path, sub_path), '=====')
    data_cnt += 1
    with open(os.path.join(prediction_path, sub_path),encoding="utf-8") as frobj:
        for i,line in enumerate(frobj):
            content = json.loads(line.strip())
            content['articleId']=str(i)
            if str(i) not in data_dict:
                if "roberta" in sub_path or "roformer" in sub_path:
                    data_dict[content['articleId']] = {
                        'sentText': content['sentText'],
                        'sentId': content['sentID'],
                        'entityMentions': content['entityMentions'],
                        'relationMentions': Counter()
                    }
                else:
                    data_dict[content['articleId']] = {
                        'sentText': content['sentText'],
                        'sentId': content['sentId'],
                        'entityMentions': content['entityMentions'],
                        'relationMentions': Counter()
                    }
            spo_list = deleteDuplicate_v1(content['relationMentions'])
            if "unilm" in sub_path or "roformer" in sub_path:
                for spo in spo_list:
                    spo_tuple =("e1start", "xx", "em2Text", spo["em2Text"], "e21start", "xx", "label", spo["label"],"em1Text", spo["em1Text"])
                    data_dict[content['articleId']]['relationMentions'][spo_tuple] += 1
            else:
                for spo in spo_list:
                    spo_tuple =("e1start", "xx", "em2Text", spo["em2Text"], "e21start",  "xx", "label", spo["label"],"em1Text", spo["em1Text"])
                    data_dict[content['articleId']]['relationMentions'][spo_tuple] += 1

print(data_cnt)
# spo_list.append({"e1start":list(spo[1])[0],"em2Text":spo[3],"e21start":list(spo[4])[0],"label":spo[2],"em1Text":spo[0]})
# wr.write(json.dumps(
#     {"articleId": "666", "sentId": "6", "entityMentions": [], "sentText": text, "relationMentions": spo_list},
#     ensure_ascii=False))
# vote = 5
vote = 2
with open(os.path.join(args.output_file, 'merge_{}_{}.json'.format(data_cnt, vote)), 'w',encoding="utf-8") as fwobj:
    for key in data_dict:
        tmp_dict = {
            'articleId':"666",
            "sentId": "6",
            "entityMentions":data_dict[key]['entityMentions'],
            'sentText':data_dict[key]['sentText'],
            'relationMentions':[]
        }
        for spo_key in data_dict[key]['relationMentions']:
            if data_dict[key]['relationMentions'][spo_key] >= int(vote):
                spo_dict = {
                    "e1start":spo_key[1], "em2Text": spo_key[3], "e21start": spo_key[5], "label":
                    spo_key[7], "em1Text": spo_key[9]
                }
                tmp_dict['relationMentions'].append(spo_dict)
        fwobj.write(json.dumps(tmp_dict, ensure_ascii=False)+'\n')