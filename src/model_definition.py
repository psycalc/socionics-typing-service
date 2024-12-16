import torch
import torch.nn as nn
from transformers import BertModel

class BertForAspects(nn.Module):
    def __init__(self, num_labels=8):
        super(BertForAspects, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.drop = nn.Dropout(0.2)  # увеличиваем дропаут до 0.2
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        dropped = self.drop(pooled_output)
        logits = self.classifier(dropped)
        return logits
