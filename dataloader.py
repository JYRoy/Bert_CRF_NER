import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import tiktoken
import numpy as np

import torch
from torch.utils.data import Dataset, random_split, DataLoader
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
MAX_LENGTH = 373
# 开始字符和结束字符
START_TAG = "<START>"
STOP_TAG = "<STOP>"
# 标签和序号的对应码表
tag_to_ix = {
    "O": 0,
    "B-dis": 1,
    "I-dis": 2,
    "B-sym": 3,
    "I-sym": 4,
    START_TAG: 5,
    STOP_TAG: 6,
}


class NERDataset(Dataset):

    def __init__(self, data_file=None):
        assert data_file != None
        self.sentences = []
        self.labels = []

        with open(data_file, "rt") as f:
            current_sentence = []
            current_labels = []
            for idx, line in enumerate(f):
                line = line.strip()
                if line:
                    word, tag = line.split("\t")
                    current_sentence.append(word)
                    current_labels.append(tag)
                else:
                    if current_sentence:
                        self.sentences.append("".join(current_sentence))
                        self.labels.append(current_labels)
                        current_sentence = []
                        current_labels = []
            if current_sentence:
                self.sentences.append("".join(current_sentence))
                self.labels.append(current_labels)

    def __len__(self):
        assert len(self.sentences) == len(self.labels)
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx], self.labels[idx]

    def get_sentence_list(self):
        return self.sentences

    def get_tag_list(self):
        return self.labels

def collate_fn_wo_bert(batch_samples):
    sents = [i[0] for i in batch_samples]
    labels = [i[1] for i in batch_samples]

    data = tokenizer.batch_encode_plus(
        batch_text_or_text_pairs=sents,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt",
        return_length=True,
    )
    
    labels_ids = []
    for label in labels:
        label_id_list = [tag_to_ix[l] for l in label]
        padding_list = [0] * (MAX_LENGTH - len(label))
        label_id_list.extend(padding_list)
        labels_ids.append(label_id_list)

    label_ids = torch.tensor(labels_ids, dtype=torch.long)
    input_ids = data["input_ids"]
    attention_mask = data["attention_mask"]
    token_type_ids = data["token_type_ids"]
    return input_ids, attention_mask, token_type_ids, label_ids

# train_dataset = NERDataset(data_file="datasets/ner_data.txt")
# print(f"train set size: {len(train_dataset)}")
# print(next(iter(train_dataset)))
# print(train_dataset.get_sentence_list())
# print(train_dataset.get_tag_list())

# train_dataloader = DataLoader(
#     train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn_wo_bert
# )
# batch = next(iter(train_dataloader))
# print(batch)
