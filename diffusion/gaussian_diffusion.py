# diffusion/gaussian_diffusion.py — Gaussian diffusion (DDPM/DDIM)
# Noise schedule, training losses, and sampling all handled here
# Includes hyperbolic distance loss extension for HiT

import math
import numpy as np
import torch as th
import enum

from .diffusion_utils import discretized_gaussian_log_likelihood, normal_kl


def mean_flat(tensor):
    """Average over all dimensions except batch."""
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


class ModelMeanType(enum.Enum):
    PREVIOUS_X = enum.auto()
    START_X = enum.auto()
    EPSILON = enum.auto()


class ModelVarType(enum.Enum):
    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()
    RESCALED_MSE = enum.auto()
    KL = enum.auto()
    RESCALED_KL = enum.auto()
    POINCARE = enum.auto()  # HiT hyperbolic distance loss

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


def _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, warmup_frac):
    """Linearly increase beta during warmup period."""
    betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    warmup_time = int(num_diffusion_timesteps * warmup_frac)
    betas[:warmup_time] = np.linspace(beta_start, beta_end, warmup_time, dtype=np.float64)
    return betas


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    """Create beta schedule by name."""
    if beta_schedule == "quad":
        betas = (
            np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "warmup10":
        betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.1)
    elif beta_schedule == "warmup50":
        betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.5)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":
        betas = 1.0 / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """Return beta schedule from string name (linear / cosine)."""
    if schedule_name == "linear":
        scale = 1000 / num_diffusion_timesteps
        return get_beta_schedule(
            "linear",
            beta_start=scale * 0.0001,
            beta_end=scale * 0.02,
            num_diffusion_timesteps=num_diffusion_timesteps,
        )
    elif schedule_name == "squaredcos_cap_v2":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """Compute beta sequence by inverting an alpha_bar function."""
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """Extract values from 1D numpy array by timestep index and broadcast."""
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


class GaussianDiffusion:
    """Gaussian diffusion process — training/sampling utilities.

    q(x_t|x_0) forward process + p(x_{t-1}|x_t) reverse process.
    Extended with hyperbolic distance loss for HiT.
    """

    def __init__(self, *, betas, model_mean_type, model_var_type, loss_type,
                 curvature=1e-5):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.curvature = curvature  # HiT Poincare ball curvature

        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        # Precompute alpha cumprod and related coefficients
        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # q(x_t|x_0) coefficients
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # Posterior q(x_{t-1}|x_t, x_0) coefficients
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        ) if len(self.posterior_variance) > 1 else np.array([])

        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod)
        )

    def q_mean_variance(self, x_start, t):
        """Mean and variance of forward process q(x_t|x_0)."""
        mean = _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """Inject noise into x_0 to produce x_t (reparameterization)."""
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """Mean and variance of posterior q(x_{t-1}|x_t, x_0)."""
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None):
        """Compute model-predicted p(x_{t-1}|x_t) mean and variance."""
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)
        model_output = model(x, t, **model_kwargs)
        if isinstance(model_output, tuple):
            model_output, extra = model_output
        else:
            extra = None

        # Learned variance: half of model output is variance interpolation value
        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = th.split(model_output, C, dim=1)
            min_log = _extract_into_tensor(self.posterior_log_variance_clipped, t, x.shape)
            max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
            # Rescale v from [-1,1] to [0,1] and interpolate between min/max
            frac = (model_var_values + 1) / 2
            model_log_variance = frac * max_log + (1 - frac) * min_log
            model_variance = th.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        # Predict x_0 from model output type
        if self.model_mean_type == ModelMeanType.START_X:
            pred_xstart = process_xstart(model_output)
        else:
            pred_xstart = process_xstart(
                self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
            )
        model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)

        assert model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
            "extra": extra,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        """Recover x_0 from noise prediction."""
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        """Recover noise from x_0 prediction."""
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    # =====================================================================
    # DDPM sampling
    # =====================================================================

    def p_sample(self, model, x, t, clip_denoised=True, denoised_fn=None,
                 cond_fn=None, model_kwargs=None):
        """Single reverse step (DDPM)."""
        out = self.p_mean_variance(
            model, x, t, clip_denoised=clip_denoised,
            denoised_fn=denoised_fn, model_kwargs=model_kwargs,
        )
        noise = th.randn_like(x)
        # No noise added at t=0
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        if cond_fn is not None:
            out["mean"] = self.condition_mean(cond_fn, out, x, t, model_kwargs=model_kwargs)
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_loop(self, model, shape, noise=None, clip_denoised=True,
                      denoised_fn=None, cond_fn=None, model_kwargs=None,
                      device=None, progress=False):
        """Full T->0 reverse sampling loop."""
        final = None
        for sample in self.p_sample_loop_progressive(
            model, shape, noise=noise, clip_denoised=clip_denoised,
            denoised_fn=denoised_fn, cond_fn=cond_fn,
            model_kwargs=model_kwargs, device=device, progress=progress,
        ):
            final = sample
        return final["sample"]

    def p_sample_loop_progressive(self, model, shape, noise=None, clip_denoised=True,
                                  denoised_fn=None, cond_fn=None, model_kwargs=None,
                                  device=None, progress=False):
        """DDPM sampling yielding each step (for visualization)."""
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.p_sample(
                    model, img, t, clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn, cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                )
                yield out
                img = out["sample"]

    # =====================================================================
    # DDIM sampling
    # =====================================================================

    def ddim_sample(self, model, x, t, clip_denoised=True, denoised_fn=None,
                    cond_fn=None, model_kwargs=None, eta=0.0):
        """Single DDIM step (deterministic when eta=0)."""
        out = self.p_mean_variance(
            model, x, t, clip_denoised=clip_denoised,
            denoised_fn=denoised_fn, model_kwargs=model_kwargs,
        )
        if cond_fn is not None:
            out = self.condition_score(cond_fn, out, x, t, model_kwargs=model_kwargs)
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        noise = th.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_prev)
            + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_loop(self, model, shape, noise=None, clip_denoised=True,
                         denoised_fn=None, cond_fn=None, model_kwargs=None,
                         device=None, progress=False, eta=0.0):
        """Full DDIM sampling loop."""
        final = None
        for sample in self.ddim_sample_loop_progressive(
            model, shape, noise=noise, clip_denoised=clip_denoised,
            denoised_fn=denoised_fn, cond_fn=cond_fn,
            model_kwargs=model_kwargs, device=device, progress=progress, eta=eta,
        ):
            final = sample
        return final["sample"]

    def ddim_sample_loop_progressive(self, model, shape, noise=None, clip_denoised=True,
                                     denoised_fn=None, cond_fn=None, model_kwargs=None,
                                     device=None, progress=False, eta=0.0):
        """DDIM sampling yielding each step."""
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.ddim_sample(
                    model, img, t, clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn, cond_fn=cond_fn,
                    model_kwargs=model_kwargs, eta=eta,
                )
                yield out
                img = out["sample"]

    # =====================================================================
    # Conditional sampling (classifier guidance)
    # =====================================================================

    def condition_mean(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """Shift mean by classifier gradient."""
        gradient = cond_fn(x, t, **model_kwargs)
        new_mean = p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float()
        return new_mean

    def condition_score(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """Apply classifier gradient to noise prediction."""
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        eps = self._predict_eps_from_xstart(x, t, p_mean_var["pred_xstart"])
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(x, t, **model_kwargs)
        out = p_mean_var.copy()
        out["pred_xstart"] = self._predict_xstart_from_eps(x, t, eps)
        out["mean"], _, _ = self.q_posterior_mean_variance(x_start=out["pred_xstart"], x_t=x, t=t)
        return out

    # =====================================================================
    # Loss functions
    # =====================================================================

    def _vb_terms_bpd(self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None):
        """Compute VLB terms (bits per dimension)."""
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.p_mean_variance(
            model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        kl = normal_kl(true_mean, true_log_variance_clipped, out["mean"], out["log_variance"])
        kl = mean_flat(kl) / np.log(2.0)
        # Use discrete log-likelihood at t=0
        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)
        output = th.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None,
                        proj_head=None):
        """Compute training loss.

        Args:
            proj_head: Projection head (4096->proj_dim), for HiT hyperbolic loss.
        """
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)

        terms = {}

        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            terms["loss"] = self._vb_terms_bpd(
                model=model, x_start=x_start, x_t=x_t, t=t,
                clip_denoised=False, model_kwargs=model_kwargs,
            )["output"]
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps

        elif self.loss_type in (LossType.MSE, LossType.RESCALED_MSE, LossType.POINCARE):
            model_output = model(x_t, t, **model_kwargs)

            # Separate learned variance + VLB auxiliary loss
            if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
                B, C = x_t.shape[:2]
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                model_output, model_var_values = th.split(model_output, C, dim=1)
                # Detach so variance prediction doesn't affect mean prediction
                frozen_out = th.cat([model_output.detach(), model_var_values], dim=1)
                terms["vb"] = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,
                    x_start=x_start, x_t=x_t, t=t, clip_denoised=False,
                )["output"]
                if self.loss_type == LossType.RESCALED_MSE:
                    terms["vb"] *= self.num_timesteps / 1000.0

            # Target selection (epsilon-prediction is default)
            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=x_start, x_t=x_t, t=t
                )[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]
            assert model_output.shape == target.shape == x_start.shape

            if self.loss_type == LossType.POINCARE:
                # HiT: hyperbolic distance loss
                poincare_loss, pred_hyp, mse_metric = self._poincare_loss(
                    model_output, target, proj_head=proj_head,
                )
                terms["mse"] = poincare_loss
                terms["poincare"] = poincare_loss
                terms["mse_monitor"] = mse_metric  # Pure MSE for monitoring only
                terms["pred_hyp"] = pred_hyp
                if self.model_mean_type == ModelMeanType.EPSILON:
                    terms["pred_x0"] = self._predict_xstart_from_eps(
                        x_t=x_t, t=t, eps=model_output
                    )
            else:
                terms["mse"] = mean_flat((target - model_output) ** 2)

            if "vb" in terms:
                terms["loss"] = terms["mse"] + terms["vb"]
            else:
                terms["loss"] = terms["mse"]
        else:
            raise NotImplementedError(self.loss_type)

        return terms

    def _poincare_loss(self, pred, target, proj_head=None):
        """Compute hyperbolic distance loss.

        1/sqrt(d) scaling then expmap0 -> lands near tanh(1)~0.762,
        preventing gradient explosion near ball boundary.

        Returns:
            (loss, pred_hyp, mse_metric): distance^2, projected prediction, pure MSE
        """
        from models.hyperbolic import PoincareBallOps
        poincare = PoincareBallOps(c=self.curvature)

        pred_flat = pred.reshape(pred.shape[0], -1)
        target_flat = target.reshape(target.shape[0], -1)

        # Original-space MSE (no gradient flow, monitoring only)
        mse_metric = mean_flat((target_flat - pred_flat) ** 2)

        # Project to low dimension if proj_head is available
        if proj_head is not None:
            pred_flat = proj_head(pred_flat)
            target_flat = proj_head(target_flat)

        # 1/sqrt(d) scaling -> tangent clip -> expmap0 -> hard clip
        scale = math.sqrt(pred_flat.shape[1])
        pred_scaled = pred_flat / scale
        target_scaled = target_flat / scale

        pred_scaled = poincare.tangent_clip(pred_scaled, alpha=0.95)
        target_scaled = poincare.tangent_clip(target_scaled, alpha=0.95)

        pred_hyp = poincare.hard_clip(poincare.expmap0(pred_scaled))
        target_hyp = poincare.hard_clip(poincare.expmap0(target_scaled))

        dist = poincare.poincare_distance(pred_hyp, target_hyp)
        return dist ** 2, pred_hyp, mse_metric

    # =====================================================================
    # Misc
    # =====================================================================

    def _prior_bpd(self, x_start):
        """Prior KL (difference from standard normal at step T)."""
        B, T = x_start.shape[0], self.num_timesteps
        t = th.tensor([T - 1] * B, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance,
            mean2=0.0, logvar2=0.0,
        )
        return mean_flat(kl_prior) / np.log(2.0)
