import torch
import torch.nn.functional as F
from tqdm import tqdm

def test(model, test_loader, device, num_classes=1):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            outputs = model(input_ids, attention_mask)
            if num_classes == 1:
                loss = F.binary_cross_entropy_with_logits(outputs.squeeze(), labels.float())
            else:
                loss = F.cross_entropy(outputs, labels)

            if num_classes == 1:
                predicted = (outputs > 0).long()
            else:
                _, predicted = torch.max(outputs, dim=1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            total_loss += loss.item()

    epoch_loss = total_loss / len(test_loader) if len(test_loader) > 0 else 0
    epoch_acc = 100 * correct / total if total > 0 else 0
    return epoch_loss, epoch_acc