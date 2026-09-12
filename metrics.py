import torch
from sklearn.metrics import recall_score, f1_score

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_num = 0
    # 新增：保存全部预测、真实标签，用来算recall、f1
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            pred = torch.argmax(logits, dim=-1)
            total_correct += torch.sum(pred == labels).item()
            total_num += labels.size(0)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    avg_loss = total_loss / len(loader)
    avg_acc = total_correct / total_num
    # 计算recall,f1
    avg_recall = recall_score(all_labels, all_preds, average="macro")
    avg_f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, avg_acc, avg_recall, avg_f1
