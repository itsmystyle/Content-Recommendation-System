import os
import pickle

import pandas as pd
import torch
from transformers import BertTokenizer

from torch.utils.data import Dataset


class MultiTaskDataset(Dataset):
    def __init__(
        self,
        data_dir,
        mode="train",
        bert_config="bert-base-multilingual-cased",
        tokenizer=None,
        test_path=None,
    ):
        self.data_dir = data_dir
        self.mode = mode
        self.bert_config = bert_config

        path = os.path.join(data_dir, "class_1_mapping.pkl")
        self.is_file_exists(path)
        with open(path, "rb") as fin:
            self.class_1_dict = pickle.load(fin)

        path = os.path.join(data_dir, "class_2_mapping.pkl")
        self.is_file_exists(path)
        with open(path, "rb") as fin:
            self.class_2_dict = pickle.load(fin)

        if mode == "test":
            path = test_path
            self.is_file_exists(path)
        elif mode == "valid":
            path = os.path.join(data_dir, "valid.csv")
            self.is_file_exists(path)
        elif mode == "train":
            path = os.path.join(data_dir, "train.csv")
            self.is_file_exists(path)
        else:
            raise Exception("Unknown mode %s" % self.mode)

        self.data = pd.read_csv(path)
        self.data = self.data.fillna("UNK")

        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            if self.bert_config.split("-")[-1] == "uncased":
                self.tokenizer = BertTokenizer.from_pretrained(self.bert_config, do_lower_case=True)
            else:
                self.tokenizer = BertTokenizer.from_pretrained(
                    self.bert_config, do_lower_case=False
                )

        self.CLS = self.tokenizer.cls_token
        self.SEP = self.tokenizer.sep_token
        self.PAD = self.tokenizer.pad_token
        self.MAX_LEN = 512

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        eventid = self.data.iloc[index].eventid
        title = self.data.iloc[index].title
        descr = self.data.iloc[index].descr
        class_1 = self.data.iloc[index].class_1
        class_2 = self.data.iloc[index].class_2

        return {
            "eventid": eventid,
            "title": title,
            "descr": descr,
            "class_1": class_1,
            "class_2": class_2,
        }

    def collate_fn(self, batch):
        # eventid
        eventid = [d["eventid"] for d in batch]

        # [CLS] descr [SEP] title
        content = [
            self.CLS
            + " "
            + d["descr"]
            + " "
            + self.SEP
            + " "
            + d["title"]
            + " "
            + self.SEP
            for d in batch
        ]
        content = [self.tokenizer.tokenize(t) for t in content]
        content_len = max([len(t) for t in content])
        max_content_len = min(self.MAX_LEN, content_len)
        content = [
            t + [self.PAD] * (max_content_len - len(t))
            if len(t) < self.MAX_LEN
            else t[: self.MAX_LEN - 1] + t[-1]
            for t in content
        ]
        content = [self.tokenizer.convert_tokens_to_ids(t) for t in content]
        content = torch.tensor(content, dtype=torch.long)

        # create a token type ids with everyting 0 by the first [SEP] (also included),
        # after which everything will be marked as 1 (padding included).
        token_type_ids = []
        for p in content:
            token_type_id = []
            type_id = 0
            for token in p:
                token_type_id.append(type_id)
                if token.item() == self.tokenizer.sep_token_id and type_id == 0:
                    type_id += 1
            token_type_ids.append(token_type_id)
        token_type_ids = torch.tensor(token_type_ids)

        # create a mask of 1s for each token followed by 0s for padding
        attention_masks = []
        for seq in content:
            seq_mask = [float(i > 0) for i in seq]
            attention_masks.append(seq_mask)
        attention_masks = torch.tensor(attention_masks)

        # class_1
        class_1 = [self.class_1_dict["item2idx"][d["class_1"]] for d in batch]
        class_1 = torch.tensor(class_1, dtype=torch.long)

        # class_2
        class_2 = [self.class_2_dict["item2idx"][d["class_2"]] for d in batch]
        class_2 = torch.tensor(class_2, dtype=torch.long)

        return {
            "eventid": eventid,
            "content": content,
            "token_type_ids": token_type_ids,
            "attention_masks": attention_masks,
            "class_1": class_1,
            "class_2": class_2,
        }

    def is_file_exists(self, path):
        if not os.path.exists(path):
            raise Exception("%s is not exist." % path)

    @property
    def num_class_1(self):
        return len(self.class_1_dict["idx2item"])

    @property
    def num_class_2(self):
        return len(self.class_2_dict["idx2item"])
