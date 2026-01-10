# src/models/__init__.py

from .vit import VisionTransformer

def build_model(cfg):
    model_cfg = cfg["model"]

    return VisionTransformer(
            num_classes = model_cfg["num_classes"],
            img_size = model_cfg["img_size"],
            patch_size = model_cfg["patch_size"],
            in_channels = model_cfg["in_channels"],
            model_dim = model_cfg["model_dim"],
            max_len = model_cfg["max_len"],
            drop = model_cfg["drop"],
            num_heads = model_cfg["num_heads"],
            mlp_ratio = model_cfg["mlp_ratio"],
            qkv_bias = model_cfg["qkv_bias"],
            num_layers = model_cfg["num_layers"],
            )
