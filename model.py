import copy
from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F
import torch
from transformers import BertTokenizer, BertModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
pretrained_model = BertModel.from_pretrained("bert-base-chinese")
pretrained_model = pretrained_model.to("cuda")


@dataclass
class ModelConfig:
    embedding_dim: int = 200


class BERT_CRF(nn.Module):

    def __init__(self, config: ModelConfig, tag_to_ix, sequence_length):
        super(BERT_CRF, self).__init__()

        self.bert = PretrainedBertModel()
        self.config = config
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        self.max_sequence_length = sequence_length
        # 转移矩阵，[tagset_size, tagset_size]
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))
        self.hidden_dim = 768  # bert embedding dim

        # 任何合法的句子不会转移到"START_TAG"，设置为-10000
        # 任何合法的句子不会从"STOP_TAG"继续转移, 设置为-10000
        self.transitions.data[self.tag_to_ix["<START>"], :] = -10000
        self.transitions.data[:, self.tag_to_ix["<STOP>"]] = -10000

        # 构建全连线性层, 一端对接BiLSTM, 另一端对接输出层, 注意输出层维度是tagset_size
        self.hidden2tag = nn.Linear(self.hidden_dim, self.tagset_size)

    def _get_bert_features(self, input_ids, attention_mask=None, token_type_ids=None):
        bert_output = self.bert(input_ids, attention_mask, token_type_ids)
        out = bert_output["last_hidden_state"]
        feats = self.hidden2tag(out)
        return feats

    def forward(self, input_ids, attention_mask, token_type_ids):
        out = self._get_bert_features(input_ids, attention_mask, token_type_ids)
        return out


class PretrainedBertModel(nn.Module):

    def __init__(self):
        super(PretrainedBertModel, self).__init__()

    def forward(self, input_ids, attention_mask, token_type_ids):

        with torch.no_grad():
            out = pretrained_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
        return out
