# diffusion/ — Gaussian diffusion module
# Includes hyperbolic distance loss extension for HiT

from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps


def create_diffusion(
    timestep_respacing,
    noise_schedule="linear",
    use_kl=False,
    sigma_small=False,
    predict_xstart=False,
    learn_sigma=True,
    rescale_learned_sigmas=False,
    diffusion_steps=1000,
    use_poincare=False,
    curvature=1e-5,
):
    """Create diffusion object — noise schedule + loss type + respacing."""
    betas = gd.get_named_beta_schedule(noise_schedule, diffusion_steps)
    if use_poincare:
        loss_type = gd.LossType.POINCARE
    elif use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if timestep_respacing is None or timestep_respacing == "":
        timestep_respacing = [diffusion_steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        curvature=curvature,
    )
