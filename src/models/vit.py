# src/models/vit.py

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """
    input: (B, C, H, W)
    output: (B, N, D)
    """
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(dim=2)  # 从第二维度开始flatten, (B, embed_dim, H' * W')
        x = x.transpose(1, 2)  # (B, H' * W', embed_dim) - (B, N, D)
        return x


class MultiHeadSelfAttention(nn.Module):
    """
    标准的自注意力
    """
    def __init__(self, model_dim, num_heads, qkv_bias, attn_drop, proj_drop):
        super().__init__()
        assert model_dim % num_heads == 0, f"model_dim:{model_dim} must be divisible by num_heads:{num_heads}"
        
        self.modle_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv_proj = nn.Linear(model_dim, model_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(p=attn_drop)
        self.proj = nn.Linear(model_dim, model_dim)
        self.proj_drop = nn.Dropout(p=proj_drop)

    def forward(self, x):
        # x: (B, N, D)
        B, N, D = x.shape

        qkv = self.qkv_proj(x)  # (B, N, 3D)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)  # (B, N, 3, num_heads, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv.unbind(dim=0)  # 3 * (B, num_heads, N, head_dim)

        attn_scores = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, N)
        attn_weights = attn_scores.softmax(dim=-1)  # (B, num_heads, N, N)
        attn_weights = self.attn_drop(attn_weights)

        # context, attn_output都可
        context = attn_weights @ v  # (B, num_heads, N, head_dim)
        out = context.transpose(1, 2).reshape(B, N, D)  # (B, N, D)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class EncoderLayer(nn.Module):
    """
    标准的pre-layernorm风格encoderlayer
    """
    def __init__(self, model_dim, num_heads, mlp_ratio, qkv_bias, attn_drop, mlp_drop):
        super().__init__()
