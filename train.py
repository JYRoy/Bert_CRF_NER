import random
import time
from tqdm.auto import tqdm

import torch
from transformers import AdamW, get_scheduler

from model import *
from dataloader import *

EPOCH_NUM = 1


# 函数sentence_map()完成中文文本信息的数字编码, 将中文语句变成数字化张量
def sentence_map(sentence_list, char_to_id, max_length):
    # 首先对一个批次的所有语句按照句子的长短进行排序, 这个操作并非必须
    sentence_list.sort(key=lambda x: len(x), reverse=True)
    # 定义一个最终存储结果特征张量的空列表
    sentence_map_list = []
    # 循环遍历一个批次内所有的语句
    for sentence in sentence_list:
        # 采用列表生成式来完成中文字符到id值的映射
        sentence_id_list = [char_to_id[c] for c in sentence]
        # 长度不够max_length的部分用0填充
        padding_list = [0] * (max_length - len(sentence))
        # 将每一个语句扩充为相同长度的张量
        sentence_id_list.extend(padding_list)
        # 追加进最终存储结果的列表中
        sentence_map_list.append(sentence_id_list)

    # 返回一个标量类型的张量
    return torch.tensor(sentence_map_list, dtype=torch.long)


def train_loop(dataloader, model, optimizer, lr_scheduler, epoch, total_loss):
    progress_bar = tqdm(range(len(dataloader)))
    progress_bar.set_description(f"loss: {0:>7f}")
    finish_batch_num = (epoch - 1) * len(dataloader)

    model.train()
    for batch, batch_data in enumerate(dataloader, start=1):
        input_ids = batch_data[0].to(device)
        attention_mask = batch_data[1].to(device)
        token_type_ids = batch_data[2].to(device)
        # labels = batch_data[3].to(device)

        logits = model(input_ids, attention_mask, token_type_ids)
        # loss = F.cross_entropy(logits, labels)

        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        # lr_scheduler.step()

        # total_loss += loss.item()
        progress_bar.set_description(
            f"loss: {total_loss/(finish_batch_num + batch):>7f}"
        )
        progress_bar.update(1)
    return total_loss


def valid_loop(dataloader, model):
    progress_bar = tqdm(range(len(dataloader)))
    all_true, all_pred = [], []
    model.eval()
    for batch, batch_data in enumerate(dataloader, start=1):
        input_ids = batch_data[0].to(device)
        attention_mask = batch_data[1].to(device)
        token_type_ids = batch_data[2].to(device)
        labels = batch_data[3].to(device)

        logits = model(input_ids, attention_mask, token_type_ids)

        pred = torch.argmax(logits, -1)
        pred_labels = pred.cpu().numpy().tolist()
        true_labels = labels.cpu().numpy().tolist()
        all_pred.extend(pred_labels)
        all_true.extend(true_labels)
        progress_bar.update(1)


train_dataset = NERDataset(data_file="datasets/ner_data.txt")
print(f"train set size: {len(train_dataset)}")
sentence_list = train_dataset.get_sentence_list()

char_to_id = {"<PAD>": 0}

max_sentence_length = 0
for sentence in sentence_list:
    if max_sentence_length < len(sentence):
        max_sentence_length = len(sentence)
    for c in sentence:
        if c not in char_to_id:
            char_to_id[c] = len(char_to_id)

print("max_sentence_length: ", max_sentence_length)
sentence_sequence = sentence_map(sentence_list, char_to_id, max_sentence_length)

train_dataloader = DataLoader(
    train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn_wo_bert
)
valid_dataloader = DataLoader(
    train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn_wo_bert
)

model = BERT_CRF(
    config=ModelConfig, tag_to_ix=tag_to_ix, sequence_length=max_sentence_length
)
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.95), eps=1e-8)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=EPOCH_NUM * len(train_dataloader),
)

total_loss = 0.0
for t in range(EPOCH_NUM):
    print(f"Epoch {t+1}/{EPOCH_NUM}\n-------------------------------")
    total_loss = train_loop(
        train_dataloader, model, optimizer, lr_scheduler, t + 1, total_loss
    )
    valid_loop(valid_dataloader, model)

print("Done!")
