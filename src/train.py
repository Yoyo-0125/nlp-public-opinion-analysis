import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

def train(model, train_loader, optimizer, device, num_classes, epoch):
    model.train()
    total_loss = 0.0
    correct, total = 0, 0
    early_stop = False

    for batch in tqdm(train_loader, desc=f"[Epoch {epoch+1}]", leave=False):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        outputs = model(input_ids, attention_mask)
        if num_classes == 1:
            loss = F.binary_cross_entropy_with_logits(outputs.squeeze(), labels.float())
        else:
            loss = F.cross_entropy(outputs, labels)

        loss.backward()
        optimizer.step()

        if num_classes == 1:
                predicted = (outputs > 0).long()
        else:
            _, predicted = torch.max(outputs, dim=1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        total_loss += loss.item()

        if loss.item() < 0.001:
            print("\nEarly Stopped: ", loss.item())
            early_stop = True
            break

    torch.save(model.state_dict(), r".\models\lstm_sentiment_small.pth")
    print("Model saved.")

    epoch_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0
    epoch_acc = 100 * correct / total if total > 0 else 0
    return epoch_loss, epoch_acc, early_stop