# src/engine/test_engine.py

import torch
import torch.nn as nn
import yaml

from src.models import build_model
from src.datasets import build_dataloader
from src.engine.train_engine import evaluate
from src.utils.vis import to_pil


def load_cfg(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run_test(cfg_path):
    cfg = load_cfg(cfg_path)
    test_cfg = cfg["test"]

    # 模型评估准备
    # device, 模型,数据加载器,损失函数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    _, _, test_loader = build_dataloader(cfg)
    criterion = nn.CrossEntropyLoss()

    ckpt_path = test_cfg["ckpt_path"]
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"[Test] Loss={test_loss:.2f}, acc={test_acc:.2f}%")

    images, labels = next(iter(test_loader))
    img = to_pil(images[0])
    img.save(test_cfg["img_save"])
