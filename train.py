import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import swanlab

from config import load_config
from dataset import ToutiaoDataset, collate_fn
from model import MineModel
from metrics import evaluate
from utils import print_config
import os

def main():
    cfg = load_config("config.json")
    print_config(cfg)
    device = cfg["device"]
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])

    # 构建数据集
    train_dataset = ToutiaoDataset(cfg["data"]["train_txt_path"], cfg["label_map"])
    dev_dataset = ToutiaoDataset(cfg["data"]["dev_txt_path"], cfg["label_map"])
    test_dataset = ToutiaoDataset(cfg["data"]["test_txt_path"], cfg["label_map"])
    print(f"Train:{len(train_dataset)}  Dev:{len(dev_dataset)}  Test:{len(test_dataset)}")

    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer, cfg["max_seq_len"])
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer, cfg["max_seq_len"])
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer, cfg["max_seq_len"])
    )

    # 模型、损失、优化器
    model = MineModel(
        model_name=cfg["model_name"],
        dropout_rate=cfg["dropout_rate"],
        num_classes=cfg["num_classes"]
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])

    #修改流程，训练完成后选取最好的dev model进行test
    best_val_acc = 0.0
    best_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Remote", "model", "best_model.pth")
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

    # swanlab
    swanlab.init(project=cfg["swanlab"]["project"])

    # 训练循环
    print("\nStarting training...")
    for epoch in range(cfg["num_epochs"]):
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
                print(f"Epoch [{epoch+1}/{cfg['num_epochs']}] | Batch [{batch_idx}/{total_steps}] | Loss:{loss.item():.4f} | AvgLoss:{step_avg_loss:.4f}")

        train_avg_loss = train_loss_sum / len(train_loader)
        # val_loss, val_acc = evaluate(model, dev_loader, criterion, device)
        val_loss, val_acc, val_recall, val_f1 = evaluate(model, dev_loader, criterion, device)

        print(
            f"==== Epoch {epoch + 1} Done ==== Train Loss:{train_avg_loss:.4f} | Val Loss:{val_loss:.4f} | Val Acc:{val_acc:.4f} | Val Recall:{val_recall:.4f} | Val F1:{val_f1:.4f}\n")

        swanlab.log({
            "train/loss": float(train_avg_loss),
            "val/loss": float(val_loss),
            "val/acc": float(val_acc),
            "val/recall": float(val_recall),
            "val/f1": float(val_f1)
        })

        #保存最优结果
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"Best Acc: {best_val_acc:.4f}\n")
    # 测试集评估
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    print("Run test set evaluation ...")

    test_loss, test_acc, test_recall, test_f1 = evaluate(model, test_loader, criterion, device)
    print(
        f"======== Test Result (Best Dev Model) ======== Test Loss:{test_loss:.4f} | Test Acc:{test_acc:.4f} | Test Recall:{test_recall:.4f} | Test F1:{test_f1:.4f}")
    swanlab.log({
        "test/loss": float(test_loss),
        "test/acc": float(test_acc),
        "test/recall": float(test_recall),
        "test/f1": float(test_f1)
    })

    swanlab.finish()
if __name__ == "__main__":
    main()
