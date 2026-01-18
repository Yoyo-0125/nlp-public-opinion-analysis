import torch.nn as nn
from transformers import BertModel

class BERTClassifier(nn.Module):
    def __init__(self, dropout=0.3, num_classes=1):
        super().__init__()
        self.bert = BertModel.from_pretrained("hfl/chinese-roberta-wwm-ext")
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS]
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        return logits