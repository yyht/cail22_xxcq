import torch
import json
import sys
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import configparser
import logging
import torch
import io
import torch.nn.functional as F
import random
import numpy as np
import time
import math
import datetime
import torch.nn as nn
import logging
from models.gpNet import RawGlobalPointer, sparse_multilabel_categorical_crossentropy
from transformers import BertTokenizerFast, BertModel,AutoModel,AutoConfig
import configparser
from models.gpNet_baffine import CoPredictor

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

import os, sys
cur_dir_path = os.path.dirname(__file__)
sys.path.extend([cur_dir_path])

# 控制台参数传入
# import argparse
# def parse_args():
#     parser = argparse.ArgumentParser(description=__doc__)
#     parser.add_argument(
#         "--input_file",
#         default=None,
#         type=str,
#         help="The config file.", )
    
#     parser.add_argument(
#         "--config_file",
#         default=None,
#         type=str,
#         help="The config file.", )
    
#     parser.add_argument(
#         "--output_file",
#         default=None,
#         type=str,
#         help="The out tag added to output_path.", )

#     args = parser.parse_args()

#     return args

# args = parse_args() # 从命令行获取

class Predict(object):
    def __init__(self, args):
        import os, sys, torch

        con = configparser.ConfigParser()
        con_path = os.path.join(cur_dir_path, args['config_file'])
        con.read(con_path, encoding='utf8')

        print(cur_dir_path, '======', con_path)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        args_path = dict(dict(con.items('paths')), **dict(con.items("para")))

        model_path = args_path['model_path']
        vocab_path = args_path['vocab_path']
        maxlen = con.getint('para', 'maxlen')
        batch_size = con.getint('para', 'batch_size')
        lr = con.getfloat('para', 'lr')
        epochs = con.getint('para', 'epochs')
        seed = 42
        use_fgm = False
        use_ema = False

        tokenizer = BertTokenizerFast.from_pretrained(vocab_path, do_lower_case=True,add_special_tokens=True)
        config = AutoConfig.from_pretrained(model_path)
        encoder = AutoModel.from_config(config)

        predicate2id = {'traffic_in': 0, 'sell_drugs_to': 1, 'posess': 2, 'provide_shelter_for': 3}
        id2predicate = {0: 'traffic_in', 1: 'sell_drugs_to', 2: 'posess', 3: 'provide_shelter_for'}

        mention_detect = CoPredictor(2, hid_size=config.hidden_size,
                                     biaffine_size=config.hidden_size,
                                     channels=config.hidden_size,
                                     ffnn_hid_size=config.hidden_size,
                                     dropout=0.1,
                                     tril_mask=True).to(device)

        s_o_head = CoPredictor(len(id2predicate), hid_size=config.hidden_size,
                               biaffine_size=config.hidden_size,
                               channels=config.hidden_size,
                               ffnn_hid_size=config.hidden_size,
                               dropout=0.1,
                               tril_mask=False).to(device)

        s_o_tail = CoPredictor(len(id2predicate), hid_size=config.hidden_size,
                               biaffine_size=config.hidden_size,
                               channels=config.hidden_size,
                               ffnn_hid_size=config.hidden_size,
                               dropout=0.1,
                               tril_mask=False).to(device)

        class ERENet(nn.Module):
            def __init__(self, encoder, a, b, c):
                super(ERENet, self).__init__()
                self.mention_detect = a
                self.s_o_head = b
                self.s_o_tail = c
                self.encoder = encoder

            def forward(self, batch_token_ids, batch_mask_ids, batch_token_type_ids):
                outputs = self.encoder(batch_token_ids, batch_mask_ids, batch_token_type_ids)

                mention_outputs = self.mention_detect(outputs, batch_mask_ids)
                so_head_outputs = self.s_o_head(outputs, batch_mask_ids)
                so_tail_outputs = self.s_o_tail(outputs, batch_mask_ids)
                return mention_outputs, so_head_outputs, so_tail_outputs

        net = ERENet(encoder, mention_detect, s_o_head, s_o_tail).to(device)
        output_path = os.path.join(cur_dir_path, args_path['output_path'])
        net.load_state_dict(torch.load(output_path))
        # net.half()
        # torch.save(net.state_dict(), "./fp1fangerenet.pth")
        # net.load_state_dict(torch.load('./fp1fangerenet.pth'))
        net.eval()

        output_file = os.path.join(cur_dir_path, args['output_file'])
        print(output_file, '===output_file===')

        from tqdm import tqdm

        text_list = []
        with open(args['input_file'], encoding="utf-8") as f, open(output_file, 'w', encoding="utf-8") as wr:
            lines = f.readlines()
            for data in (lines):
                data = json.loads(data.strip())['sentText']
                text_list.append(data)
            # ids=[json.loads(text.rstrip())["ID"] for text in f.readlines()]
            for text in tqdm(text_list):
                mapping = tokenizer(text.lower(), return_offsets_mapping=True, max_length=maxlen,add_special_tokens=True)["offset_mapping"]
                threshold = 0.0
                encoder_txt = tokenizer.encode_plus(text.lower(), max_length=512)
                input_ids = torch.tensor(encoder_txt["input_ids"]).long().unsqueeze(0).to(device)
                token_type_ids = torch.tensor(encoder_txt["token_type_ids"]).unsqueeze(0).to(device)
                attention_mask = torch.tensor(encoder_txt["attention_mask"]).unsqueeze(0).to(device)
                with torch.no_grad():
                    scores = net(input_ids, attention_mask, token_type_ids)
                outputs = [o[0].data.cpu().numpy() for o in scores]
                subjects, objects = set(), set()
                outputs[0][:, [0, -1]] -= np.inf
                outputs[0][:, :, [0, -1]] -= np.inf
                for l, h, t in zip(*np.where(outputs[0] > 0)):
                    if l == 0:
                        subjects.add((h, t))
                    else:
                        objects.add((h, t))
                spoes = set()
                for sh, st in subjects:
                    for oh, ot in objects:
                        p1s = np.where(outputs[1][:, sh, oh] > threshold)[0]
                        p2s = np.where(outputs[2][:, st, ot] > threshold)[0]
                        ps = set(p1s) & set(p2s)
                        for p in ps:
                            spoes.add((
                                text[mapping[sh][0]:mapping[st][-1]], (mapping[sh][0], mapping[st][-1]),
                                id2predicate[p],
                                text[mapping[oh][0]:mapping[ot][-1]], (mapping[oh][0], mapping[ot][-1])
                            ))
                spo_list = []
                for spo in list(spoes):
                    spo_list.append({"e1start":list(spo[1])[0],"em2Text":spo[3],"e21start":list(spo[4])[0],"label":spo[2],"em1Text":spo[0]})
                    # spo_list.append({'h': {'name': spo[0], 'pos': list(spo[1])}, 't': {'name': spo[3], 'pos': list(spo[4])}, 'relation': spo[2]})
                wr.write(json.dumps({"articleId":"666","sentId":"6","entityMentions":[],"sentText":text, "relationMentions":spo_list}, ensure_ascii=False))
                wr.write("\n")
                spo_list = []