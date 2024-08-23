# Bert_CRF_NER
Chinese NER using BERT and CRF by PyTorch

## CRF

CRF 使用 BERT 的发射矩阵作为输入，输出预测的标注序列。

CRF 的**损失函数定义**为：

$$
 -log P(\bar{y}|x) = -log\frac{e^{score(x, \bar{y})}}{\sum_y{e^{score(x, y)}}} = -(score(x, \bar{y}) - log(\sum_y e^{score(x, y)}))
$$

损失函数可以认为包含两个部分：

1. 单条真实路径的分数：$score(x, \bar{y})$
2. 全部路径的分数构成的归一化项：$log(e^{s_1} + e^{s_2} + ... + e^{s_N})$

具体的**每条路径的分数定义**为：

$$
score(x, y) = \sum_{i=1}^n P_{w_i, y_i} + \sum_{i = 0}^n A_{y_{i},y_{i+1}}
$$

- $P_{w_i,y_i}$：发射矩阵，每行对应一个单词的发射分数，每列代表一个标签，每个单元表示当前行的单词到当前列的标签的分数，发射矩阵大小为[seq_len, tag_size]
- $A_{y_{i},y_{i+1}}$：转移矩阵，表示标签之间相互转移的分数，转移矩阵大小为[tag_size, tag_size]

## Viterbi 解码

Viterbi 算法和前向算法类似，都是在全部路径中动态规划地找到最优路径，具体地说是选择每个位置累计地最大路径。

Viterbi 解码解决的问题可以被描述为：

给定长度为 n 的序列，标签数为 m（标签值表示为 1,2,....,m），发射概率矩阵 E（n * m），其中 E[i][j] 表示第 i 个词预测为标签 j 的发射概率，转移概率矩阵 T(m*m)，其中 T[i][j] 为标签 j 转移到标签 i 的转移概率。要求返回最优的序列标注结果（一个数组 res，res[i] 表示第 i 个词标注的标签值）。

### 回溯过程

解码过程依赖于一个核心对象：

- backponinters：[当前句子单词数+1，tag数目]，对于一个单词来说，记录的是取对应下标时的 tag 来自于上一个单词的哪一个 tag

举一个例子：

```shell
当前 Tag                       0, 1, 2, 3, 4, 5, 6   0, 1, 2, 3, 4, 5, 6    0, 1, 2, 3, 4, 5, 6    0, 1, 2, 3, 4, 5, 6    0, 1, 2, 3, 4, 5, 6
当前 Tag 的上一个单词的 Tag   [[5, 5, 5, 5, 5, 5, 5], [4, 1, 1, 4, 1, 6, 1], [4, 1, 1, 4, 1, 6, 4], [4, 0, 1, 4, 1, 6, 4], [3, 0, 1, 3, 1, 6, 3]]
```

这个 list 就是 backponinters，记录了第一个到最后一个单词的 tag 回溯列表。

在 backponinters 的构建过程中，我们会记录一个 best_tag_id，在最后计算完转移到 STOP_TAG 的分数后，我们取最大值对应的 idx，即为最后一个单词转移到 STOP_TAG 的 tag id。

```python
# 最后加上其他tag转移到STOP_TAG的分数
terminal_var = forward_var + self.transitions[self.tag_to_ix["<STOP>"]]
best_tag_id = argmax(terminal_var)
```

假设 best_tag_id 为 0，根据以下代码进行回溯：

```python
for bptrs_t in reversed(backpointers):
    # 通过第i个time_step得到的最佳id, 找到第i-1个time_step的最佳id
    best_tag_id = bptrs_t[best_tag_id]
    best_path.append(best_tag_id)
```

可以获得的 best_path 为：（这个序列不一定对，只是解释过程）

```shell
[5,1,1,4,3]
```

### 前向过程

从上面的代码中看出，backpointers 和 best_tag_id 是在前向过程中构建的。

前向过程依赖一个核心对象：

- **forward_var**：[1, tag 数目]，表示单词被标注为各个 tag 的分数，计算过程依赖的核心公式如下，即为上一个单词被标注为各个 tag 的分数 + 到当前单词的当前 tag 的转移分数。

```python
current_tag_var = forward_var + self.transitions[current_tag]
```

根据 current_tag_var 获取上一个单词的 tag 到当前单词的当前 tag 的最大分数的 tag。

```python
best_tag_id = argmax(current_tag_var)
```

将转移到当前 current_tag 的最大的分数和上一个单词的 tag id 记录下来

```python
bptrs_t.append(best_tag_id)
# 最大的得分也保存下来
viterbivars_t.append(current_tag_var[0][best_tag_id].view(1))
```

上述的整个过程会进行 7 次，因为有 7 个 tag，分别求出当前单词取 7 个不同 tag 的情况。

对于每一个单词，我们可以获取到一个 bptrs_t，shape 是 [1, 7]，代表当前单词为对应列的 tag 时最有可能（分数最大的）来自于上一个单词的哪一个 tag。

在获取 best_tag_id 同时，为了能够进行前向计算，我们还要不停的更新 forward_var，这里要注意的是，在更新时要使用发射分数

```python
for current_tag in range(self.tagset_size):
    ...
    viterbivars_t.append(current_tag_var[0][best_tag_id].view(1))
forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
```


