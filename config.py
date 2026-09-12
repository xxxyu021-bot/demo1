import json
import os
import torch


def load_config(config_path="config.json"):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # 自动处理device
    if cfg["device"] == "cuda":
        cfg["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 拼接绝对路径
    root_dir = os.path.dirname(os.path.abspath(__file__))
    cfg["data"]["train_txt_path"] = os.path.join(root_dir, cfg["data"]["train_txt_path"])
    cfg["data"]["dev_txt_path"] = os.path.join(root_dir, cfg["data"]["dev_txt_path"])
    cfg["data"]["test_txt_path"] = os.path.join(root_dir, cfg["data"]["test_txt_path"])

    # label_map key转回int
    cfg["label_map"] = {int(k): v for k, v in cfg["label_map"].items()}
    return cfg
