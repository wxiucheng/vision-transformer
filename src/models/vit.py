# src/models/vit.py

import math
import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """
    input: (B, C, H, W)
    output: (B, N, D)
    """
    def __init__(self, img_size, patch_size, in_channels, model_dim):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size

        self.proj = nn.Conv2d(in_channels, model_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0, f"H{H}, W{W} must be divisible by patch_size, "
        x = self.proj(x)  # (B, model_dim, H/patch_size, W/patch_size)
        x = x.flatten(2)  # 从第二维度开始flatten, (B, model_dim, H' * W')
        x = x.transpose(1, 2)  # (B, H' * W', model_dim) - (B, N, D)
        return x


class MLP(nn.Module):
    """
    fc1 + act + drop + fc2 + drop
    """
    def __init__(self, in_features, hidden_features, drop):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.drop = nn.Dropout(p=drop)
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        # x: (B, N, D)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        
        return x

class MultiHeadSelfAttention(nn.Module):
    """
    标准的自注意力
    """
    def __init__(self, model_dim, num_heads, qkv_bias, drop):
        super().__init__()
        assert model_dim % num_heads == 0, f"model_dim:{model_dim} must be divisible by num_heads:{num_heads}"
        
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv_proj = nn.Linear(model_dim, model_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(p=drop)
        self.proj = nn.Linear(model_dim, model_dim)
        self.proj_drop = nn.Dropout(p=drop)

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
    def __init__(self, model_dim, num_heads, mlp_ratio, qkv_bias, drop):
        super().__init__()

        self.norm1 = nn.LayerNorm(model_dim)
        self.attn = MultiHeadSelfAttention(
                model_dim = model_dim,
                num_heads = num_heads,
                qkv_bias = qkv_bias,
                drop = drop,
                )
        self.norm2 = nn.LayerNorm(model_dim)
        self.mlp = MLP(
                in_features = model_dim,
                hidden_features = int(model_dim * mlp_ratio),
                drop = drop,
                )

    def forward(self, x):
        # x: (B, N, D)
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        
        return x


class Encoder(nn.Module):
    """
    标准的Encoder
    """
    def __init__(self, num_layers, model_dim, num_heads, mlp_ratio, qkv_bias, drop):
        super().__init__()

        self.encoder = nn.Sequential(*[
            EncoderLayer(
                model_dim = model_dim,
                num_heads = num_heads,
                mlp_ratio = mlp_ratio,
                qkv_bias = qkv_bias,
                drop = drop,
                )
            for _ in range(num_layers)
            ])

    def forward(self, x):
        return self.encoder(x)


class PositionalEncoding1D(nn.Module):
    """
    标准的1D正余弦位置编码
    """
    def __init__(self, model_dim, max_len, drop):
        super().__init__()
        assert model_dim % 2 == 0, f"model_dim{model_dim} must be divisible by 2"

        self.model_dim = model_dim
        self.max_len = max_len
        self.drop = nn.Dropout(p=drop)

        # 位置索引
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(dim=1)  # (max_len, 1)

        # 频率项
        div_term = torch.exp(
                torch.arange(0, model_dim, 2, dtype=torch.float32)
                * (-math.log(10000.0) / model_dim)
                )  # (model_dim/2, )

        # 位置编码
        pe = torch.zeros(max_len, model_dim, dtype=torch.float32)  # (max_len, model_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # register_buffer
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        # x: (B, N, D)
        B, N, D = x.shape
        if N > self.max_len:
            raise ValueError(f"N > max_len{self.max_len}")
        
        return self.drop(x + self.pe[:, :N, :])

class VisionTransformer(nn.Module):
    """
    切分patch + 加上CLS token + 位置编码(带dropout) + 多个EncoderLayer + 取出CLS做分类 + 分类头
    """
    def __init__(self, num_classes, img_size, patch_size, in_channels, model_dim, max_len, drop, num_heads, mlp_ratio, qkv_bias, num_layers):
        super().__init__()

        self.patch_embedding = PatchEmbedding(
                img_size = img_size,
                patch_size = patch_size,
                in_channels = in_channels,
                model_dim = model_dim,
                )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, model_dim))  # (1, 1, model_dim)

        self.positional_encoding1d = PositionalEncoding1D(
                model_dim = model_dim,
                max_len = max_len,
                drop = drop,
                )

        self.encoder = Encoder(
                num_layers = num_layers,
                model_dim = model_dim,
                num_heads = num_heads,
                mlp_ratio = mlp_ratio,
                qkv_bias = qkv_bias,
                drop = drop,
                )

        self.classifier = nn.Linear(model_dim, num_classes)

    def forward_features(self, x):
        # x: (B, 3, H, W)
        x = self.patch_embedding(x)  # (B, N, D)

        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # (B, 1, model_dim)

        x = torch.cat((cls_token, x), dim=1)  # (B, N + 1, D)

        x = self.positional_encoding1d(x)  # (B, N + 1, D)

        x = self.encoder(x)  # (B, N + 1, D)

        return x[:, 0]  # CLS token feature: (B, D)

    def forward(self, x):
        # x: (B, 3, H, W)
        x = self.forward_features(x)  # (B, D)
        x = self.classifier(x)  # (B, num_classes)
        return x

def main():
    # ====== 超参数（小一点，方便跑）======
    num_classes = 10
    img_size = 224
    patch_size = 16
    in_channels = 3
    model_dim = 192
    num_heads = 3
    mlp_ratio = 4.0
    qkv_bias = True
    num_layers = 4
    drop = 0.1

    # 关键：max_len 必须 >= num_patches + 1
    num_patches = (img_size // patch_size) ** 2
    max_len = num_patches + 1

    model = VisionTransformer(
        num_classes=num_classes,
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        model_dim=model_dim,
        max_len=max_len,
        drop=drop,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        num_layers=num_layers,
    )

    model.eval()

    # ====== 构造输入 ======
    x = torch.randn(2, 3, img_size, img_size)  # B=2

    # ====== 前向 ======
    with torch.no_grad():
        logits = model(x)

    print("logits shape:", logits.shape)  # 期望 (2, num_classes)
    assert logits.shape == (2, num_classes)
    print("[OK] forward pass works.")

if __name__ == "__main__":
    main()
