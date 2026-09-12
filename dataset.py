import torch
from torch.utils.data import Dataset
import os

class ToutiaoDataset(Dataset):

    def __init__(self, txt_path, label_map):
        self.txt_path = txt_path
        self.label_map = label_map
        self.samples = []

        self._load_data()

    # 读取数据封装成一个函数当成员函数，init调用
    def _load_data(self):
        with open(self.txt_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("_!_")
                raw_label_code = int(parts[1])
                text = parts[3]
                label = self.label_map[raw_label_code]
                self.samples.append({"text": text, "label": label})


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch, tokenizer, max_seq_len):
    texts = [item["text"] for item in batch]
    labels = [item["label"] for item in batch]
    encode = tokenizer(
        texts,
        padding="longest",
        truncation=True,
        max_length=max_seq_len,
        return_tensors="pt"
    )
    encode["labels"] = torch.tensor(labels, dtype=torch.long)
    return encode
