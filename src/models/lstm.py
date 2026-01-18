import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hid_dim, num_layers, num_classes, pad_idx):
        super().__init__()
        self.pad_idx = pad_idx
        self.num_classes = num_classes
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            input_size=embed_dim, 
            hidden_size=hid_dim, 
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(hid_dim*2, num_classes) # because bidirectional=True
        # self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

        
    def forward(self, text, attention_mask=None):
        x = self.embedding(text)
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_hidden_state = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1) # because bidirectional
        output = self.fc(last_hidden_state).squeeze(1)
        if self.num_classes != 1:
            output = self.sigmoid(output)
        return output