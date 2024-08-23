import random
import time
from tqdm.auto import tqdm

import torch
from transformers import AdamW, get_scheduler

from model import *
from dataloader import *

EPOCH_NUM = 100


def train_loop(dataloader, model, optimizer, lr_scheduler, epoch, total_loss):
    progress_bar = tqdm(range(len(dataloader)))
    progress_bar.set_description(f"loss: {0:>7f}")
    finish_batch_num = (epoch - 1) * len(dataloader)

    model.train()
    for batch, batch_data in enumerate(dataloader, start=1):
        optimizer.zero_grad()
        input_ids = batch_data[0]
        attention_mask = batch_data[1]
        token_type_ids = batch_data[2]
        labels = batch_data[3]

        loss = model.neg_log_likelihood(
            input_ids, attention_mask, token_type_ids, labels
        )

        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        result = model(input_ids)
        print(result)

        total_loss += loss.item()
        progress_bar.set_description(
            f"loss: {total_loss/(finish_batch_num + batch):>7f}"
        )
        progress_bar.update(1)
    return total_loss


train_dataset = NERDataset(data_file="datasets/ner_data_mini.txt")
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

train_dataloader = DataLoader(
    train_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn_wo_bert
)
valid_dataloader = DataLoader(
    train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn_wo_bert
)

model = BERT_CRF(
    config=ModelConfig, tag_to_ix=tag_to_ix, sequence_length=max_sentence_length
)
# model = model.to(device)
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

print("Done!")
