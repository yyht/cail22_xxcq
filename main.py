

from predict_cail22_xxcq_roformer_negative_learning import Predict as nlba_predict
from predict_seq2struct_unilm_cail22_xxcq import Predict as unilm_predict
from predict_baffine import Predict as baffine_predict
import configparser

import os, sys
cur_dir_path = os.path.dirname(__file__)
sys.path.extend([cur_dir_path])

con = configparser.ConfigParser()
con_path = os.path.join(cur_dir_path, './my_config/roformer_negative_learning.ini')
con.read(con_path, encoding='utf8')

args_path = dict(dict(con.items('paths')), **dict(con.items("para")))

nlba_predict(args_path)

con = configparser.ConfigParser()
con_path = os.path.join(cur_dir_path, './my_config/unilm_roformer_large.ini')
con.read(con_path, encoding='utf8')

args_path = dict(dict(con.items('paths')), **dict(con.items("para")))

unilm_predict(args_path)

# con = configparser.ConfigParser()
# con_path = os.path.join(cur_dir_path, './my_config/unilm_large_v6.ini')
# con.read(con_path, encoding='utf8')

# args_path = dict(dict(con.items('paths')), **dict(con.items("para")))

# unilm_predict(args_path)


con = configparser.ConfigParser()
con_path = os.path.join(cur_dir_path, './my_config/baffine_v1.ini')
con.read(con_path, encoding='utf8')

args_path = dict(dict(con.items('paths')), **dict(con.items("para")))

baffine_predict(args_path)

con = configparser.ConfigParser()
con_path = os.path.join(cur_dir_path, './my_config/baffine_v2.ini')
con.read(con_path, encoding='utf8')

args_path = dict(dict(con.items('paths')), **dict(con.items("para")))

baffine_predict(args_path)

import os
import os
from collections import Counter

import os, sys
from collections import Counter
cur_dir_path = os.path.dirname(__file__)
sys.path.extend([cur_dir_path])

prediction_path = os.path.join(cur_dir_path, 'final_test')

data_dict = {}

from functools import reduce
import json
def deleteDuplicate_v1(input_dict_lst):
    f = lambda x,y:x if y in x else x + [y]
    return reduce(f, [[], ] + input_dict_lst)

valid_sub_path = [
    'baffine_v2.json',
    'baffine_v1.json',
    'roformer_negative_learning.json',
    'unilm_roformer_v4.json'
]

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
vote = 2
with open('/result/result.json', 'w',encoding="utf-8") as fwobj:
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


