from transformers import MegatronBertConfig, MegatronBertModel
from transformers import BertTokenizer
from transformers import BertTokenizerFast, BertModel, BertConfig

# tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext-large")
# config = MegatronBertConfig.from_pretrained("IDEA-CCNL/Erlangshen-MegatronBert-1.3B")
# model = MegatronBertModel.from_pretrained("IDEA-CCNL/Erlangshen-MegatronBert-1.3B")

# model = BertModel.from_pretrained("hfl/chinese-roberta-wwm-ext-large")
# config = BertConfig.from_pretrained("hfl/chinese-roberta-wwm-ext-large")

from roformer import RoFormerModel, RoFormerConfig  
tokenizer = BertTokenizerFast.from_pretrained('junnyu/roformer_v2_chinese_char_base', do_lower_case=True)
model = RoFormerModel.from_pretrained('junnyu/roformer_v2_chinese_char_base')
# print(encoder)
config = RoFormerConfig.from_pretrained('junnyu/roformer_v2_chinese_char_base')

config.save_pretrained('/mnt/sijinghui.sjh/BERT/roformer_base_v2')
model.save_pretrained('/mnt/sijinghui.sjh/BERT/roformer_base_v2')
tokenizer.save_pretrained('/mnt/sijinghui.sjh/BERT/roformer_base_v2')