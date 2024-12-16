import torch

def accuracy_fn(logits, labels):
    preds = torch.argmax(logits, dim=1)
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    return correct / total

def save_model(model, path):
    torch.save(model.state_dict(), path)

def multi_label_accuracy_fn(logits, labels, threshold=0.5):
    # logits: [batch, 8]
    # labels: [batch, 8]
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    correct = (preds == labels).float().mean().item()  # доля совпадений
    return correct
