import copy
import logging
import time
from typing import Optional, List, Dict

import torch
from botorch.acquisition import qLogNoisyExpectedImprovement
from botorch.sampling import SobolQMCNormalSampler
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.kernels import AdditiveKernel

from gtbo.gaussian_process import (
    robust_optimize_acqf,
    test_mlls_and_return_best_state_dict,
    fit_mll,
    get_gp,
)


def create_candidates(
    x: torch.Tensor,
    fx: torch.Tensor,
    device: str,
    remove_first_samples: int = 0,
    active_dimensions: Optional[List[int]] = None,
    model_hyperparameters: Optional[Dict] = None,
    prev_mll: Optional[ExactMarginalLogLikelihood] = None,
):
    """
    Create candidate points for the next batch.
    This entails fitting a GP model to the current data and maximizing the acquisition function to find the next batch.

    Args:
            remove_first_samples: The number of samples to remove from the beginning of the dataset
            prev_mll: The previous marginal log likelihood
            active_dimensions: The active dimensions
            x: The current points in the trust region
            fx: The function values at the current points
            device: The device to use
            model_hyperparameters: The hyperparameters of the GP model

    Returns:
        The candidate points and the corresponding function values and the model hyperparameters, gp fitting time, ei maximization time

    """
    # Scale the function values
    mean = torch.mean(fx)
    std = torch.std(fx)
    fx_scaled = (fx - mean) / std

    # qnei doesn't support minimization
    fx_scaled = -fx_scaled

    if device == "cuda":
        fx_scaled = fx_scaled.to(device)
        x = x.to(device)

    get_gp_kwargs = dict(
        active_dimensions=active_dimensions,
        remove_first_samples=remove_first_samples,
    )
    model, train_x, _ = get_gp(x=x, fx=fx_scaled, **get_gp_kwargs)

    gp_fitting_time_start = time.time()

    model_hyperparameters, mll = fit_mll(
        model=model,
        model_hyperparameters=model_hyperparameters,
    )

    if prev_mll is not None:
        # clone mlls
        mll = copy.deepcopy(mll)
        prev_mll = copy.deepcopy(prev_mll)

        best_state_dict = test_mlls_and_return_best_state_dict(
            new_mll=mll.to(device),
            prev_mll=prev_mll.to(device),
        )

        # if the new mll is better, we use it, otherwise we use the old one
        model.load_state_dict(best_state_dict)

    logging.debug(f"Mean AFTER: {model.mean_module.constant.item():.3f}")
    gp_fitting_time_end = time.time()
    gp_fitting_time = gp_fitting_time_end - gp_fitting_time_start

    acq_opt_time_start = time.time()

    # use qNoisyExpectedImprovement
    sampler = SobolQMCNormalSampler(1024)
    acq = qLogNoisyExpectedImprovement(
        model=model,
        X_baseline=train_x,
        sampler=sampler,
        maximize=False,
        prune_baseline=True,
    )

    x_cand_ret, vals = robust_optimize_acqf(
        acq_function=acq,
        bounds=torch.tensor(
            [[0.0] * train_x.shape[1], [1.0] * train_x.shape[1]],
            dtype=torch.double,
            device=device,
        ),
        q=1,
        options={"sample_around_best": True, "batch_limit": 2048},
    )

    loss = -vals
    acq_opt_time_end = time.time()
    acq_opt_time = acq_opt_time_end - acq_opt_time_start

    if isinstance(model.covar_module.base_kernel, AdditiveKernel):
        hp_summary = torch.cat(
            (
                model.covar_module.outputscale.reshape(1, 1),
                model.likelihood.noise_covar.noise.reshape(1, 1),
                model.covar_module.base_kernel.kernels[0].lengthscale.reshape(1, -1),
                model.covar_module.base_kernel.kernels[1].lengthscale.reshape(1, -1),
            ),
            dim=1,
        )
    else:
        hp_summary = torch.cat(
            (
                model.covar_module.outputscale.reshape(1, 1),
                model.likelihood.noise_covar.noise.reshape(1, 1),
                model.covar_module.base_kernel.lengthscale.reshape(1, -1),
            ),
            dim=1,
        )
    return (
        x_cand_ret,
        loss,
        hp_summary,
        model_hyperparameters,
        gp_fitting_time,
        acq_opt_time,
        mll,
    )
