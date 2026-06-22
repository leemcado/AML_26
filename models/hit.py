# models/hit.py — Hyperbolic Diffusion Transformer
# Extends DiT: hyperbolic distance loss induces hierarchical structure in latents
# forward() produces the same Euclidean output as DiT (sampling/VAE compatible)
# Hyperbolic projection handled externally by proj_head + _poincare_loss

from functools import partial

from .dit import DiT, SIZE_CONFIGS


class HiT(DiT):
    """Hyperbolic Diffusion Transformer.

    Architecture identical to DiT; only the loss function differs (hyperbolic distance).
    proj_head (Stiefel) projects to low-dim, then expmap0 maps to hyperbolic space.
    """

    def __init__(self, *args, curvature: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)


# =========================================================================
# Model configs
# =========================================================================

def build_hit(
    size: str = "B", patch_size: int = 2, curvature: float = 1.0, **kwargs
) -> HiT:
    """Build HiT from size + patch_size + curvature."""
    if size not in SIZE_CONFIGS:
        raise ValueError(f"Unknown size '{size}'. Options: {list(SIZE_CONFIGS.keys())}")
    return HiT(patch_size=patch_size, curvature=curvature, **SIZE_CONFIGS[size], **kwargs)


# Registry for eval script compatibility
HiT_models = {
    f"HiT-{size}/{ps}": partial(build_hit, size=size, patch_size=ps)
    for size in ("S", "B", "L")
    for ps in (2, 4, 8)
}
