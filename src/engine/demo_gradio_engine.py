# src/engine/demo_gradio_engine.py

import os

import yaml
import torch
import gradio as gr

from src.models import build_model
from src.datasets import CIFAR10Transforms


def load_cfg(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_model_and_transform(cfg, ckpt_path, device):
    model = build_model(cfg).to(device)

    if os.path.exists(ckpt_path):
        raise FileNotFoundError(f"ckpt_path{ckpt_path} not found")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    class_names = ckpt["class_names"]

