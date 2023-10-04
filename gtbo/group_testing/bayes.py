# coding=utf-8
# Copyright 2020 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Major code changes happened here

# Lint as: python3
"""Computes the prior, posterior, and likelihood."""
import math
from typing import Dict

import torch
import numpy as np

from gtbo.util import logit


def log_likelihood(
    particles: torch.Tensor,
    test_results: torch.Tensor,
    groups: torch.Tensor,
    noise_variance: float,
    signal_variance: float,
    dtype: torch.dtype = torch.double,
    device: str = "cpu",
) -> torch.Tensor:
    """Computes individual (parallel) log_likelihood of k_groups test results.

    Args:
        noise_variance: the variance of the noise in the test results.
        signal_variance: the variance of the signal in the test results.
        particles: torch.Tensor<bool>[n_particles, n_patients]. Each one is a possible scenario of a disease status of n patients.
        test_results: torch.Tensor<bool>[n_groups] the results given by the wet lab for each of the tested groups.
        groups: torch.Tensor<bool>[num_groups, num_patients] the definition of the group that were tested.
        dtype: the type of the torch tensors.
        device: the device to run the computation on.

    Returns: The log likelihood of the particles given the test results.
    """
    positive_in_groups = groups.numpy() @ particles.T.numpy() > 0
    positive_in_groups = torch.tensor(positive_in_groups, dtype=dtype)

    n_groups = groups.shape[0]
    n_particles = particles.shape[0]

    variances = positive_in_groups * (signal_variance - noise_variance) + noise_variance

    RHT_active = np.log(2 * math.pi * signal_variance)
    RHT_inactive = np.log(2 * math.pi * noise_variance)
    RHTs = positive_in_groups * RHT_active + (1 - positive_in_groups) * RHT_inactive

    if n_groups == 0:
        return torch.tensor(0.0, dtype=dtype, device=device)

    logpdf = (
        -torch.divide(
            torch.pow(test_results, 2)
            .reshape(n_groups, -1)
            .repeat_interleave(n_particles, dim=1),
            2 * variances,
        )
        - RHTs / 2
    )

    return torch.sum(logpdf, dim=0)


def log_prior(
    particles: torch.Tensor, base_infection_rate: torch.Tensor
) -> torch.Tensor:
    """Computes log of prior probability of state using infection rate."""
    # here base_infection can be either a single number per patient or an array
    if (
        isinstance(base_infection_rate, float) or len(base_infection_rate) == 1
    ):  # only one rate
        return torch.sum(particles, axis=-1) * logit(
            base_infection_rate
        ) + particles.shape[0] * math.log(1 - base_infection_rate)
    elif base_infection_rate.shape[0] == particles.shape[-1]:  # prior per patient
        return torch.sum(
            particles * torch.logit(base_infection_rate)[torch.newaxis, :]
            + torch.log(1 - base_infection_rate)[torch.newaxis, :],
            dim=-1,
        )
    else:
        raise ValueError("Vector of prior probabilities is not of correct size")


def log_probability(
    particles: torch.Tensor,
    test_results: torch.Tensor,
    groups: torch.Tensor,
    prior_infection_rate: torch.Tensor,
    noise_variance: float,
    signal_variance: float,
    dtype: torch.dtype = torch.double,
    device: str = "cpu",
):
    """Given past tests and prior, outputs unnormalized log-probabilities.

    Args:
      device: torch.device, where to run the computation.
      dtype: torch.dtype, type of the computation.
      signal_variance: float, variance of the signal (the variance of random function values).
      noise_variance: float, variance of the noise (the variance when evaluating the function on the same point).
      particles: torch.Tensor, computing tempered log_posterior for each of these.
      test_results: torch.Tensor, probability depends on recorded test results
      groups: torch.Tensor ... tests above defined using these groups.
      prior_infection_rate: torch.Tensor, prior on infection.

    Returns:
      a vector of log probabilities
    """

    log_prob = torch.zeros(len(particles), dtype=dtype)
    if test_results is not None:
        # if sampling from scratch, include prior and rescale temperature.
        log_prob += log_likelihood(
            particles,
            test_results,
            groups,
            noise_variance,
            signal_variance,
            dtype=dtype,
            device=device,
        )
    if prior_infection_rate is not None:
        log_prob += log_prior(particles, prior_infection_rate)
    return log_prob.squeeze()


def tempered_logpos_logbase(
    particles: torch.Tensor,
    log_posterior_params: Dict[str, torch.Tensor],
    log_base_measure_params: Dict[str, torch.Tensor],
    temperature: float,
    dtype: torch.dtype = torch.double,
    device: str = "cpu",
):
    """Computes a tempered log posterior and adds a base measure."""
    lp_p = log_probability(
        particles, dtype=dtype, device=device, **log_posterior_params
    )
    lp_b = log_probability(
        particles, dtype=dtype, device=device, **log_base_measure_params
    )
    return temperature * lp_p + lp_b
