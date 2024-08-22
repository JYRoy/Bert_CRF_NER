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
