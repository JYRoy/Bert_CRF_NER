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
        # tansitions[i, j] 代表第 j 个 tag 转移到第 i 个 tage 的转移分数（转移概率），列表转行表
        # 初始化随机参数，训练出来的
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))
        self.hidden_dim = 768  # bert embedding dim

        # 任何合法的句子不会转移到"START_TAG"，设置为-10000
        # 任何合法的句子不会从"STOP_TAG"继续转移, 设置为-10000
        self.transitions.data[self.tag_to_ix["<START>"], :] = -10000
        self.transitions.data[:, self.tag_to_ix["<STOP>"]] = -10000

        # 构建全连线性层, 一端对接bert, 另一端对接输出层, 注意输出层维度是tagset_size
        # 将 bert 提取的特征向量映射到 tag 的空间，单词对应 tag 的发射概率
        self.hidden2tag = nn.Linear(self.hidden_dim, self.tagset_size)

    def _get_bert_features(self, input_ids, attention_mask=None, token_type_ids=None):
        # 通过 bert 提取特征
        bert_output = self.bert(input_ids, attention_mask, token_type_ids)
        out = bert_output["last_hidden_state"]  # [batch_size, seq_len, embed_size]
        feats = self.hidden2tag(out)  # [batch_size, seq_len, tagset_size]
        return feats

    def _forward_alg(self, feats):
        # 前向算法
        # feats 表示发射矩阵，bert 输出的句子中每个单词对应每个 tag 的发射概率

        # 用 -10000 来填充 [1, tagset_size] 大小的 tensor
        init_alphas = torch.full((1, self.tagset_size), -10000)
        # 初始化 START 位置的发射概率，<START> 的 tag 是 5，只有下标 5 的位置是 0，其他位置都是 -10000
        # 表示从 START 开始
        init_alphas[0][self.tag_to_ix["<START>"]] = 0

        forward_var = init_alphas

        result = torch.zeros((1, feats.size(0)))
        idx = 0

        for feat_line in feats:  # 遍历每一个句子
            for feat in feat_line:  # 遍历句子中每一个单词

                # 当前单词的前向 tensor
                alphas_t = []

                for next_tag in range(self.tagset_size):
                    # 取出当前单词对应当前 tag 的发射分数
                    emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)

                    # 取出当前 tag 由之前的 tag 转移过来的分数
                    trans_score = self.transitions[next_tag].view(1, -1)

                    # 当前路径的分数 = 之前时间步（单词）分数 + 转移分数 + 发射分数
                    next_tag_var = forward_var + trans_score + emit_score

                    # 对当前分数取 log—sum-exp
                    alphas_t.append(log_sum_exp(next_tag_var).view(1))

                # 更新用于存储之前时间步的分数，用于下一个时间步的分数计算
                forward_var = torch.cat(alphas_t).view(1, -1)

            # 最终要转移到 STOP tag
            terminal_var = forward_var + self.transitions[self.tag_to_ix["<STOP>"]]

            # 计算最终分数
            alpha = log_sum_exp(terminal_var)
            result[0][idx] = alpha
            idx += 1
        return result

    def _score_sentence(self, feats, tags):
        # CRF 的输出，emit + transition scores

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
        # CRF 损失函数由两部分构成，真实路径的分数和所有路径的总分数
        # 真实路径的分数是所有路径中分数最高的
        # log 真实路径的分数 / log 所有可能路径的分数，越大越好，构造 CRF loss 函数取反，loss 越小越好
        feats = self._get_bert_features(input_ids, attention_mask, token_type_ids)
        # 前向算法分数
        forward_score = self._forward_alg(feats)
        # 真实分数
        gold_score = self._score_sentence(feats, tags)

        return torch.sum(forward_score - gold_score, dim=1)

    def _viterbi_decode(self, feats):
        # decoding：给定一个已知的观测序列，求其最有可能对应的状态序列

        # 最佳路径的存放列表
        result_best_path = []

        for feat_line in feats:  # 遍历一个句子

            # 预测序列的得分
            backpointers = []

            # 初始化 viterbi 的 previous 变量
            # 约束了合法的序列只能从 START_TAG 开始
            init_vvars = torch.full((1, self.tagset_size), -10000)
            init_vvars[0][self.tag_to_ix["<START>"]] = 0

            # 将初始化的变量赋值给 forward_var
            # 在第 i 个时间步中, forward_var 保存的是第 i-1 个时间步的 viterbi 变量
            forward_var = init_vvars
            # 遍历每一个时间步
            for feat in feat_line:
                # 保存当前时间步的回溯指针
                bptrs_t = []
                # 保存当前时间步的 viterbi 变量
                viterbivars_t = []
                # 遍历所有可能的转移标签
                for next_tag in range(self.tagset_size):
                    # viterbi 算法记录最优路径时只考虑上一步的分数
                    # 以及上一步的 tag 转移到当前 tag 的转移分数
                    # 不考虑当前 tag 的发射分数
                    # forward_var 保存的是之前的最优路径的值
                    next_tag_var = forward_var + self.transitions[next_tag]
                    # 找到最大值对应的 tag
                    best_tag_id = argmax(next_tag_var)
                    bptrs_t.append(best_tag_id)
                    viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
                # 此处再将发射矩阵的分数feat添加上来, 继续赋值给forward_var, 作为下一个time_step的前向传播变量
                forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
                # 将当前time_step的回溯指针添加进当前样本行的总体回溯指针中
                backpointers.append(bptrs_t)

            # 最后加上转移到STOP_TAG的分数
            terminal_var = forward_var + self.transitions[self.tag_to_ix["<STOP>"]]
            best_tag_id = argmax(terminal_var)

            # 根据回溯指针, 解码最佳路径
            best_path = [best_tag_id]
            # 从后向前回溯最佳路径
            for bptrs_t in reversed(backpointers):
                # 通过第i个time_step得到的最佳id, 找到第i-1个time_step的最佳id
                best_tag_id = bptrs_t[best_tag_id]
                best_path.append(best_tag_id)

            # 将START_TAG去除掉
            start = best_path.pop()

            # 确认一下最佳路径的第一个标签是START_TAG
            assert start == self.tag_to_ix["<START>"]

            # 因为是从后向前进行回溯, 所以在此对列表进行逆序操作得到从前向后的真实路径
            best_path.reverse()
            # 将当前这一行的样本结果添加到最终的结果列表中
            result_best_path.append(best_path)

        return result_best_path

    def forward(self, sentence):
        bert_feats = self._get_bert_features(sentence)

        result_sequence = self._viterbi_decode(bert_feats)
        return result_sequence


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
