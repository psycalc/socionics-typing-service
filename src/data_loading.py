import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

class AspectDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        labels_str = self.labels[idx]  # например, "0 3"
        # Разбиваем по пробелам
        aspects = labels_str.split()
        label_tensor = torch.zeros(8)  # 8 аспектов
        for asp in aspects:
            label_tensor[int(asp)] = 1.0

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
            padding='max_length',
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': label_tensor
        }

def load_aspect_data(train_path="data/aspect_train.csv", val_path="data/aspect_val.csv", batch_size=16):
    train_df = pd.read_csv(train_path, dtype={"labels": str})
    val_df = pd.read_csv(val_path, dtype={"labels": str})

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    train_dataset = AspectDataset(
        texts=train_df["text"].tolist(),
        labels=train_df["labels"].tolist(),
        tokenizer=tokenizer
    )

    val_dataset = AspectDataset(
        texts=val_df["text"].tolist(),
        labels=val_df["labels"].tolist(),
        tokenizer=tokenizer
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, tokenizer
