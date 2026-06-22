# models/dit.py — Diffusion Transformer (adaLN-Zero)

import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================================
# Core blocks (patch embedding, attention, MLP)
# =========================================================================

class PatchEmbed(nn.Module):
    """Convert image to patch tokens via a single Conv2d."""

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 4,
        embed_dim: int = 768,
        bias: bool = True,
    ):
        super().__init__()
        self.patch_size = (patch_size, patch_size)
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size, bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) -> (B, num_patches, embed_dim)
        return self.proj(x).flatten(2).transpose(1, 2)


class Attention(nn.Module):
    """Multi-head self-attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        # scaled dot-product (PyTorch 2.0+)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class Mlp(nn.Module):
    """Two-layer MLP (GELU)."""

    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))



def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """adaLN modulation: x * (1 + scale) + shift."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# =========================================================================
# Positional embeddings
# =========================================================================

def _sincos_1d(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    """1D sincos positional embedding."""
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / (10000 ** omega)
    out = np.einsum("m,d->md", pos.ravel(), omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def sincos_2d_pos_embed(embed_dim: int, grid_size: int) -> np.ndarray:
    """2D sincos positional embedding (square grid)."""
    gh = np.arange(grid_size, dtype=np.float32)
    gw = np.arange(grid_size, dtype=np.float32)
    grid = np.stack(np.meshgrid(gw, gh), axis=0).reshape(2, 1, grid_size, grid_size)
    h_emb = _sincos_1d(embed_dim // 2, grid[0])
    w_emb = _sincos_1d(embed_dim // 2, grid[1])
    return np.concatenate([h_emb, w_emb], axis=1)


# =========================================================================
# Conditioning embeddings (timestep + class)
# =========================================================================

class TimestepEmbedder(nn.Module):
    """Sincos timestep embedding -> MLP projection."""

    def __init__(self, hidden_size: int, freq_dim: int = 256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    @staticmethod
    def sinusoidal_embedding(
        t: torch.Tensor, dim: int, max_period: int = 10000
    ) -> torch.Tensor:
        """Sincos frequency embedding."""
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.sinusoidal_embedding(t, self.freq_dim))


class LabelEmbedder(nn.Module):
    """Class label embedding + null token dropout for CFG."""

    def __init__(self, num_classes: int, hidden_size: int, dropout_prob: float):
        super().__init__()
        use_cfg = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def _drop_labels(
        self,
        labels: torch.Tensor,
        force_drop_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Randomly replace labels with the null token."""
        if force_drop_ids is None:
            drop_ids = (
                torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
            )
        else:
            drop_ids = force_drop_ids == 1
        return torch.where(drop_ids, self.num_classes, labels)

    def forward(
        self,
        labels: torch.Tensor,
        train: bool,
        force_drop_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Replace labels with null token at dropout_prob during training."""
        if (train and self.dropout_prob > 0) or (force_drop_ids is not None):
            labels = self._drop_labels(labels, force_drop_ids)
        return self.embedding_table(labels)


# =========================================================================
# Transformer block
# =========================================================================

class DiTBlock(nn.Module):
    """adaLN-Zero transformer block.

    Predicts 6 modulation params (shift/scale/gate) from conditioning (t+y),
    applied to attention+MLP. Zero-initialized so blocks start as identity.
    """

    def __init__(
        self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, **kwargs
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
        )
        # (shift, scale, gate) x (attn, mlp) = 6 params
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class FinalLayer(nn.Module):
    """Final adaLN + Linear (patch tokens -> output channels)."""

    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm(x), shift, scale)
        return self.linear(x)


# =========================================================================
# DiT model
# =========================================================================

class DiT(nn.Module):
    """Diffusion Transformer — patch-based latent diffusion backbone.

    Injects timestep+class conditioning via adaLN-Zero with built-in CFG dropout.
    """

    def __init__(
        self,
        input_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 4,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        class_dropout_prob: float = 0.1,
        num_classes: int = 1000,
        learn_sigma: bool = True,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size

        self.x_embedder = PatchEmbed(
            input_size, patch_size, in_channels, hidden_size, bias=True
        )
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, hidden_size), requires_grad=False
        )

        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)]
        )
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self._init_weights()

    def _init_weights(self) -> None:
        """Weight initialization — zero-init adaLN modulation outputs."""

        def _xavier(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_xavier)

        grid_size = int(self.x_embedder.num_patches ** 0.5)
        pos_embed = sincos_2d_pos_embed(self.hidden_size, grid_size)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # adaLN-Zero: all modulation outputs init to 0 -> blocks start as identity
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """Patch tokens -> spatial tensor."""
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(x.shape[0], c, h * p, h * p)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """(B,C,H,W) noisy latent -> noise prediction."""
        x = self.x_embedder(x) + self.pos_embed
        t = self.t_embedder(t)
        y = self.y_embedder(y, self.training)
        c = t + y
        for block in self.blocks:
            x = block(x, c)
        x = self.final_layer(x, c)
        return self.unpatchify(x)

    def forward_with_cfg(
        self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor, cfg_scale: float
    ) -> torch.Tensor:
        """CFG sampling: linear combination of cond/uncond outputs."""
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        guided_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([guided_eps, guided_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


# =========================================================================
# Model configs
# =========================================================================

SIZE_CONFIGS = {
    "S": dict(depth=12, hidden_size=384, num_heads=6),
    "B": dict(depth=12, hidden_size=768, num_heads=12),
    "L": dict(depth=24, hidden_size=1024, num_heads=16),
    "XL": dict(depth=28, hidden_size=1152, num_heads=16),
}


def build_dit(size: str = "B", patch_size: int = 2, **kwargs) -> DiT:
    """Build DiT from size + patch_size."""
    if size not in SIZE_CONFIGS:
        raise ValueError(f"Unknown size '{size}'. Options: {list(SIZE_CONFIGS.keys())}")
    return DiT(patch_size=patch_size, **SIZE_CONFIGS[size], **kwargs)


DiT_models = {
    f"DiT-{size}/{ps}": partial(build_dit, size=size, patch_size=ps)
    for size in SIZE_CONFIGS
    for ps in (2, 4, 8)
}
