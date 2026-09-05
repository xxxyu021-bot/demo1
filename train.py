import sys
print(sys.executable)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, AutoTokenizer
import swanlab
import os

#全局配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

current_script_dir = os.path.dirname(os.path.abspath(__file__))

project_root = current_script_dir

dropout_rate = 0.3

# # 本地bert‑base‑chinese模型路径
# 数据集txt路径
model_name = "bert-base-chinese"
train_txt_path = os.path.join(project_root, "Remote", "data", "train.txt")
dev_txt_path = os.path.join(project_root, "Remote", "data", "dev.txt")
test_txt_path = os.path.join(project_root, "Remote", "data", "test.txt")

batch_size = 16
num_epochs = 4
learning_rate = 1e-5
num_classes = 15
max_seq_len = 128

tokenizer = AutoTokenizer.from_pretrained(model_name)

label_map = {
    100: 0,   # 民生故事
    101: 1,   # 文化
    102: 2,   # 娱乐
    103: 3,   # 体育
    104: 4,   # 财经
    106: 5,   # 房产
    107: 6,   # 汽车
    108: 7,   # 教育
    109: 8,   # 科技
    110: 9,   # 军事
    112: 10,  # 旅游
    113: 11,  # 国际
    114: 12,  # 证券股票
    115: 13,  # 农业
    116: 14   # 电竞游戏
}

#数据集类
class ToutiaoDataset(Dataset):
    def __init__(self, txt_path):
        self.samples = []
        with open(txt_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("_!_")
                raw_label_code = int(parts[1])
                text = parts[3]
                label = label_map[raw_label_code]
                self.samples.append({"text": text, "label": label})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]



def collate_fn(batch):
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


#BERT分类模型
class MineModel(nn.Module):
    def __init__(self, model_name,dropout_rate):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)

        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        pooled_output = outputs.pooler_output

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

#评估函数（验证/测试集共用）
def evaluate(model, loader):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_num = 0
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
    avg_loss = total_loss / len(loader)
    avg_acc = total_correct / total_num
    return avg_loss, avg_acc

#数据加载
print("Starting training preparation...")
print(f"Model path: {model_name}")

train_dataset = ToutiaoDataset(train_txt_path)
dev_dataset = ToutiaoDataset(dev_txt_path)
test_dataset = ToutiaoDataset(test_txt_path)
print(f"Train:{len(train_dataset)}  Dev:{len(dev_dataset)}  Test:{len(test_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

#初始化
# model = MineModel(model_name).to(device)

model = MineModel(model_name, dropout_rate=dropout_rate).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)

# SwanLab初始化
# swanlab.init(project="bert-toutiao-demo1", mode="local")
swanlab.init(project="bert-toutiao-demo1")

#训练循环
print("\nStarting training...")
for epoch in range(num_epochs):
    model.train()
    train_loss_sum = 0.0
    total_steps = len(train_loader)

    for batch_idx, batch in enumerate(train_loader, 1):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item()

        if batch_idx % 20 == 0 or batch_idx == total_steps:
            step_avg_loss = train_loss_sum / batch_idx
            print(f"Epoch [{epoch+1}/{num_epochs}] | Batch [{batch_idx}/{total_steps}] | Loss:{loss.item():.4f} | AvgLoss:{step_avg_loss:.4f}")

    train_avg_loss = train_loss_sum / len(train_loader)
    val_loss, val_acc = evaluate(model, dev_loader)
    print(f"==== Epoch {epoch+1} Done ==== Train Loss:{train_avg_loss:.4f} | Val Loss:{val_loss:.4f} | Val Acc:{val_acc:.4f}\n")

    # 记录可视化日志
    # print(f"Type of train_avg_loss: {type(train_avg_loss)}, value: {train_avg_loss}")

    swanlab.log({
        "train/loss": float(train_avg_loss),
        "val/loss": float(val_loss),
        "val/acc": float(val_acc)
    })

#测试集最终评估
print("Run test set evaluation ...")
test_loss, test_acc = evaluate(model, test_loader)
print(f"======== Test Result ======== Test Loss:{test_loss:.4f} | Test Acc:{test_acc:.4f}")
# swanlab.log({"test_loss": float(test_loss), "test_acc": float(test_acc)})
swanlab.log({
    "test/loss": float(test_loss),
    "test/acc": float(test_acc)
})

swanlab.finish()

