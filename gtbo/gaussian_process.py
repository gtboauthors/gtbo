import logging
import time
import warnings
from typing import Tuple, Optional, List, Any

import gin
import gpytorch
import torch
from botorch import fit_gpytorch_mll
from botorch.acquisition import AcquisitionFunction
from botorch.exceptions import ModelFittingError
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.optim.fit import fit_gpytorch_mll_scipy, fit_gpytorch_mll_torch
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import AdditiveKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.priors import GammaPrior
from torch import Tensor
from torch.quasirandom import SobolEngine

from gtbo.priors import CustomLogNormalPrior


@gin.configurable()
def get_gp(
    x: Tensor,
    fx: Tensor,
    active_prior_parameters: Tuple[float, float],
    inactive_prior_parameters: Tuple[float, float] = None,
    active_dimensions: Optional[List[int]] = None,
    remove_first_samples: int = 0,
    gp_noise_var_estimate: Optional[float] = None,
) -> Tuple[SingleTaskGP, Tensor, Tensor]:
    """
    Define the GP model.

    Args:
            gp_noise_var_estimate: the estimate of the noise standard deviation from the GT phase
            active_prior_parameters: the prior for dimensions with active lengthscales
            inactive_prior_parameters: the prior for dimensions with inactive lengthscales
            active_dimensions: the dimensions with active lengthscales
            remove_first_samples: the number of samples to remove from the gp fitting
            x: the input points
            fx: the function values at the input points

    Returns:
        the GP model, the input points, and the function values at the input points

    """
    num_dims = x.shape[-1]
    noise_prior = None
    botorch_prior = False

    train_x = x[remove_first_samples:].detach().clone()
    train_fx = fx[remove_first_samples:, None].detach().clone()

    if gp_noise_var_estimate is not None:
        logging.warning("Ignoring noise prior since noise estimate is provided")
        _noise_prior = None
    else:
        _noise_prior = noise_prior

    # Define the model
    warnings.simplefilter("ignore")
    likelihood = (
        GaussianLikelihood(noise_constraint=GreaterThan(1e-6), noise_prior=_noise_prior)
        if not botorch_prior
        else None
    )

    if likelihood is not None and gp_noise_var_estimate is not None:
        logging.debug(f"Setting noise to {gp_noise_var_estimate:.3f}")
        likelihood.noise = gp_noise_var_estimate
        likelihood.raw_noise.requires_grad = False
    elif botorch_prior and gp_noise_var_estimate is not None:
        logging.warning("Ignoring noise estimate since botorch prior is used")

    if (
        active_dimensions is not None
    ):  # active_dimensions is not None only when we run pseudo-saasbo
        # type than lognormal
        active_mu, active_sigma = active_prior_parameters
        inactive_mu, inactive_sigma = inactive_prior_parameters
        mu, sigma = torch.zeros((1, num_dims)).to(x), torch.zeros((1, num_dims)).to(x)
        for dim in range(num_dims):
            if dim in active_dimensions:
                mu[:, dim] = active_mu
                sigma[:, dim] = active_sigma
            else:
                mu[:, dim] = inactive_mu
                sigma[:, dim] = inactive_sigma

        # Stack the mus and sigmas for the various lengthscales

        lengthscale_prior = CustomLogNormalPrior(loc=mu, scale=sigma)
        ard_size = x.shape[-1]
        outputscale_prior = GammaPrior(concentration=2, rate=0.15)

        base_kernel = gpytorch.kernels.MaternKernel(
            lengthscale_prior=lengthscale_prior,
            ard_num_dims=ard_size,
            nu=2.5,
        )

        covar_module = gpytorch.kernels.ScaleKernel(
            base_kernel,
            outputscale_prior=outputscale_prior,
        )

        model = SingleTaskGP(
            train_X=train_x,
            train_Y=train_fx,
            covar_module=covar_module,
            likelihood=likelihood,
        )
    else:
        model = SingleTaskGP(
            train_X=train_x,
            train_Y=train_fx,
            likelihood=likelihood,
        )

    return model, train_x, train_fx


@gin.configurable
def fit_mll(
    model: SingleTaskGP,
    max_cholesky_size: int = 1000,
    model_hyperparameters: Optional[dict] = None,
) -> tuple[dict[str, Any], ExactMarginalLogLikelihood]:
    """
    Fit the GP model. If the LBFGS optimizer fails, use the Adam optimizer.

    Args:
            model: the GP model
            max_cholesky_size: the maximum size of the Cholesky decomposition
            model_hyperparameters: the hyperparameters of the model

    Returns:
        the Gram matrix and the model state dictionary

    """

    if model_hyperparameters is not None:
        # Set model hyperparameters
        model.load_state_dict(model_hyperparameters)
    # Set model to training mode
    then = time.time()

    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    with gpytorch.settings.max_cholesky_size(max_cholesky_size):
        lbgs_failed = False
        if isinstance(model.covar_module.base_kernel, AdditiveKernel):
            initial_lengthscale_active = (
                model.covar_module.base_kernel.kernels[0].lengthscale.detach().clone()
            )
            initial_lengthscale_inactive = (
                model.covar_module.base_kernel.kernels[1].lengthscale.detach().clone()
            )
        else:
            initial_lengthscale = (
                model.covar_module.base_kernel.lengthscale.detach().clone()
            )
        try:
            fit_gpytorch_mll(mll=mll, optimizer=fit_gpytorch_mll_scipy)
        except (ModelFittingError, ValueError) as e:
            logging.warning(
                f"⚠ Failed to fit GP using LBFGS. Printing error, outputscale, noise, and lengthscales."
            )
            lbgs_failed = True
            if isinstance(model.covar_module.base_kernel, AdditiveKernel):
                model.covar_module.base_kernel.kernels[
                    0
                ].lengthscale = initial_lengthscale_active
                model.covar_module.base_kernel.kernels[
                    1
                ].lengthscale = initial_lengthscale_inactive
            else:
                model.covar_module.base_kernel.lengthscale = initial_lengthscale

        if lbgs_failed:
            if lbgs_failed:
                logging.warning(
                    f"⚠ Failed to fit GP using LBFGS, using backup Adam optimizer"
                )
            fit_gpytorch_mll(mll=mll, optimizer=fit_gpytorch_mll_torch)

    now = time.time()
    logging.info(f"GP fitting time: {now - then:.2f} seconds")

    return model.state_dict(), mll


@gin.configurable
def robust_optimize_acqf(
    acq_function: AcquisitionFunction,
    bounds: Tensor,
    q: int,
    num_restarts: int = 10,
    raw_samples: int = 512,
    return_best_only: bool = True,
    **kwargs,
):
    """
    Optimize the acquisition function. If the optimization fails, sample raw_samples points from the bounds and evaluate
    the acquisition function on them.

    .. warning:: This might not work for q>1.

    Args:
        acq_function: the acquisition function to optimize
        bounds: the bounds of the input space
        q: the number of points to optimize
        num_restarts: the number of restarts for the optimization
        raw_samples: the number of raw samples to use (points to evaluate before starting gradient descent)
        return_best_only: whether to return only the best point or all the points evaluated
        **kwargs: additional arguments for the optimization

    Returns:
        the best point and the value of the acquisition function at the best point (if return_best_only is True)
        or the points and the values of the acquisition function at the points (if return_best_only is False)



    """
    # TODO this works for q=1, not sure if it works for q>1
    try:
        return optimize_acqf(
            acq_function=acq_function,
            bounds=bounds,
            q=q,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            **kwargs,
        )
    except Exception as e:
        logging.warning(f"⚠ Failed to optimize acquisition function. Printing error.")
        logging.warning(e)

        # now we just Sobol sample raw_samples points from the bounds and evaluate the acquisition function on them

        points = (
            SobolEngine(q, scramble=True)
            .draw(raw_samples)
            .to(dtype=bounds.dtype, device=bounds.device)
        )
        points = points * (bounds[1] - bounds[0]) + bounds[0]
        values = acq_function(points)

        # check if acq_function has attribute 'maximize'
        maximize = True
        if hasattr(acq_function, "maximize"):
            maximize = acq_function.maximize

        # check if return_best_only is True

        best = values.max(dim=0)[0] if maximize else values.min(dim=0)[0]
        best_index = values.argmax(dim=0) if maximize else values.argmin(dim=0)
        best_point = points[best_index].reshape(q, -1) if return_best_only else points
        return best_point, best


def test_mlls_and_return_best_state_dict(
    new_mll: ExactMarginalLogLikelihood,
    prev_mll: ExactMarginalLogLikelihood,
) -> dict[str, Any]:
    """
    Test two MLLs and return the state dictionary of the best one.

    Args:
        new_mll: the new MLL
        prev_mll: the previous MLL

    Returns:
        the state dictionary of the model with the highest likelihood
    """
    if not new_mll.training:
        new_mll.train()
    if not prev_mll.training:
        prev_mll.train()

    model1 = new_mll.model
    model2 = prev_mll.model
    try:
        mll_val_1 = new_mll(model1(model1.train_inputs[0]), model1.train_targets)
        mll_val_2 = prev_mll(model2(model2.train_inputs[0]), model2.train_targets)
    except RuntimeError as e:
        logging.warning(
            f"⚠ Failed to compare MLLs. Printing error. Returning new model."
        )
        logging.warning(f"Error: '{e}'")
        return model1.state_dict()
    if mll_val_1 > mll_val_2:
        return model1.state_dict()
    else:
        return model2.state_dict()
