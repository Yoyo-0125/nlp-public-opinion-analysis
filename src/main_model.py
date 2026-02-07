import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForMaskedLM
from dataset import SentimentDataset, read_dataset, get_dataloader
from models.bert import BERTClassifier
from models.lstm import LSTMClassifier
from train import train
from test import test



df = read_dataset('..\data\weibo_senti_100k.csv')
# df = df.sample(n=10000, random_state=42)

print('Dataset size:', len(df))

train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
test_df, val_df = train_test_split(temp_df, test_size=0.2, random_state=42, stratify=temp_df['label'])

tokenizer = AutoTokenizer.from_pretrained(r'D:\Projects\nlp-sentiment-opinion-analysis\src\models\chinese-roberta-wwm-ext')

EPOCHS = 30
LR = 2e-5
BATCH_SIZE = 1024

train_loader = get_dataloader(train_df, tokenizer, batch_size=BATCH_SIZE, shuffle=True)
test_loader = get_dataloader(test_df, tokenizer, batch_size=BATCH_SIZE, shuffle=False)
val_loader = get_dataloader(val_df, tokenizer, batch_size=BATCH_SIZE, shuffle=False)

model = LSTMClassifier(
    vocab_size=tokenizer.vocab_size, 
    embed_dim=128, 
    hid_dim=64, 
    num_layers=4, 
    num_classes=1,
    pad_idx=tokenizer.pad_token_id
).to(device)
optimizer = AdamW(model.parameters(), lr=LR)

# for param in model.bert.parameters():
#     param.requires_grad = False

for epoch in range(EPOCHS):
    train_loss, train_acc, early_break = train(model, train_loader, optimizer, device, num_classes=1, epoch=epoch)
    val_loss, val_acc = test(model, val_loader, device, num_classes=1)
    print(f"[Epoch {epoch+1}/{EPOCHS}]: "
        f"Train loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% "
        f"Val loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    if early_break:
        break
