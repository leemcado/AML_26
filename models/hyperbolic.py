# models/hyperbolic.py — Poincare ball operations
# Lightweight implementation optimized for training loops, inspired by geoopt

import math
import torch


class PoincareBallOps:
    """Poincare ball model operations.

    B_c^n = {x in R^n : c * ||x||^2 < 1}, curvature -c, radius 1/sqrt(c).
    """

    def __init__(self, c=1.0):
        self.c = c

    def expmap0(self, v):
        """Exponential map at origin: tangent vector -> point on the ball."""
        sqrt_c = math.sqrt(self.c)
        v_norm = torch.norm(v, dim=-1, keepdim=True).clamp(min=1e-10)
        coeff = torch.tanh(sqrt_c * v_norm) / (sqrt_c * v_norm)
        return coeff * v

    def logmap0(self, y):
        """Logarithmic map at origin: point on the ball -> tangent vector."""
        sqrt_c = math.sqrt(self.c)
        y_norm = torch.norm(y, dim=-1, keepdim=True).clamp(min=1e-10)
        scaled_norm = (sqrt_c * y_norm).clamp(max=1.0 - 1e-5)
        coeff = torch.arctanh(scaled_norm) / (sqrt_c * y_norm)
        return coeff * y

    def tangent_clip(self, v, alpha: float = 0.95):
        """Clip tangent vector norm before expmap0 — constrain within alpha of ball boundary."""
        sqrt_c = math.sqrt(self.c)
        max_norm = 2.0 * math.atanh(alpha) / sqrt_c
        norm = torch.norm(v, dim=-1, keepdim=True).clamp(min=1e-15)
        scale = torch.clamp(max_norm / norm, max=1.0)
        return scale * v

    def hard_clip(self, x, margin: float = 4e-3):
        """L2 norm clipping to ensure point stays inside the ball (geoopt-style, margin=4e-3)."""
        norm = torch.norm(x, dim=-1, keepdim=True)
        max_norm = (1.0 - margin) / math.sqrt(self.c)
        scale = torch.clamp(max_norm / (norm + 1e-15), max=1.0)
        return x * scale

    def poincare_distance(self, x, y):
        """Hyperbolic distance between two points."""
        diff_norm_sq = torch.sum((x - y) ** 2, dim=-1)
        x_norm_sq = torch.sum(x ** 2, dim=-1)
        y_norm_sq = torch.sum(y ** 2, dim=-1)

        denom = (1.0 - self.c * x_norm_sq) * (1.0 - self.c * y_norm_sq)
        denom = denom.clamp(min=1e-10)

        arg = 1.0 + 2.0 * self.c * diff_norm_sq / denom
        arg = arg.clamp(min=1.0 + 1e-7)

        return torch.acosh(arg)

    def distance_to_origin(self, x):
        """Hyperbolic distance to origin: d(0,x) = 2/sqrt(c) * arctanh(sqrt(c)*||x||)."""
        sqrt_c = math.sqrt(self.c)
        x_norm = torch.norm(x, dim=-1)
        scaled_norm = (sqrt_c * x_norm).clamp(max=1.0 - 1e-5)
        return (2.0 / sqrt_c) * torch.arctanh(scaled_norm)

    def mobius_add(self, x, y):
        """Mobius addition x + y (used for geodesic interpolation)."""
        x_norm_sq = torch.sum(x ** 2, dim=-1, keepdim=True)
        y_norm_sq = torch.sum(y ** 2, dim=-1, keepdim=True)
        xy_dot = torch.sum(x * y, dim=-1, keepdim=True)

        num = (1.0 + 2.0 * self.c * xy_dot + self.c * y_norm_sq) * x + \
              (1.0 - self.c * x_norm_sq) * y
        denom = 1.0 + 2.0 * self.c * xy_dot + self.c ** 2 * x_norm_sq * y_norm_sq
        denom = denom.clamp(min=1e-10)

        return num / denom

    def geodesic(self, x, y, t):
        """Geodesic interpolation x->y (t in [0,1]): logmap0 -> scale -> expmap0."""
        neg_x = -x
        xy = self.mobius_add(neg_x, y)
        v = self.logmap0(xy)
        v_scaled = t * v
        xy_t = self.expmap0(v_scaled)
        return self.mobius_add(x, xy_t)
