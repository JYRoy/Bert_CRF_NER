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


@dataclass
class ModelConfig:
    embedding_dim: int = 200


# 添加几个辅助函数, 为log_sum_exp()服务
def to_scalar(var):
    # 返回一个python float类型的值
    return var.view(-1).data.tolist()[0]


def argmax(vec):
    # 返回列的维度上最大值的下标, 而且下标是一个标量float类型
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)


def log_sum_exp(vec):
    # 求向量中的最大值
    max_score = vec[0, argmax(vec)]
    # 构造一个最大值的广播变量
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    # 先减去最大值, 再求解log_sum_exp, 最终的返回值上再加上max_score
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


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

        # 构建全连线性层, 一端对接bert, 另一端对接输出层, 注意输出层维度是tagset_size
        self.hidden2tag = nn.Linear(self.hidden_dim, self.tagset_size)

    def _get_bert_features(self, input_ids, attention_mask=None, token_type_ids=None):
        bert_output = self.bert(input_ids, attention_mask, token_type_ids)
        out = bert_output["last_hidden_state"]  # [batch_size, seq_len, embed_size]
        feats = self.hidden2tag(out)  # [batch_size, seq_len, tagset_size]
        return feats

    def _forward_alg(self, feats):
        init_alphas = torch.full((1, self.tagset_size), -10000)
        init_alphas[0][self.tag_to_ix["<START>"]] = 0

        forward_var = init_alphas

        # feats = feats.transpose(1, 0)

        result = torch.zeros((1, feats.size(0)))
        idx = 0

        for feat_line in feats:  # 遍历每一个句子
            for feat in feat_line:  # 遍历句子中每一个单词
                alphas_t = []

                for next_tag in range(self.tagset_size):
                    emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)
                    trans_score = self.transitions[next_tag].view(1, -1)
                    next_tag_var = forward_var + trans_score + emit_score
                    alphas_t.append(log_sum_exp(next_tag_var).view(1))
                forward_var = torch.cat(alphas_t).view(1, -1)
            terminal_var = forward_var + self.transitions[self.tag_to_ix["<STOP>"]]
            alpha = log_sum_exp(terminal_var)
            result[0][idx] = alpha
            idx += 1
        return result

    def _score_sentence(self, feats, tags):
        # 假设有转移矩阵A，A(i, j)代表tag_i转移到tag_i的概率
        # 假设有发射矩阵P，P(i, j)代表单词w_i映射到tag_i的非归一化概率
        # 假设有n个单词
        # 损失函数中的第一项，n个单词的A(i, i+1) + n个单词的P(i, i+1)
        # [batch_size, seq_len, embed_size]

        # 初始化一个0值的tensor，为后续的累加做准备
        score = torch.zeros(1)
        # 要在tags矩阵的第一列添加,这一列全部都是START_TAG
        temp = torch.tensor(
            torch.full((feats.size(0), 1), self.tag_to_ix["<START>"]), dtype=torch.long
        )
        tags = torch.cat((temp, tags), dim=1)

        result = torch.zeros((1, feats.size(0)))
        idx = 0

        for feat_line in feats:
            # 这里是在真实标签tags的指导下进行的转移矩阵和发射矩阵的累加分数求和
            for i, feat in enumerate(feat_line):
                score = (
                    score
                    + self.transitions[tags[idx][i + 1], tags[idx][i]]
                    + feat[tags[idx][i + 1]]
                )
            score = score + self.transitions[self.tag_to_ix["<STOP>"], tags[idx][-1]]
            result[0][idx] = score
            idx += 1
        return result

    def neg_log_likelihood(self, input_ids, attention_mask, token_type_ids, tags):
        feats = self._get_bert_features(input_ids, attention_mask, token_type_ids)

        forward_score = self._forward_alg(feats)

        gold_score = self._score_sentence(feats, tags)

        return torch.sum(forward_score - gold_score, dim=1)


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
