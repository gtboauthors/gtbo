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
"""SMC Invariant Kernels / Proposals.

Follows more or less https://arxiv.org/pdf/1101.6037.pdf in addition to Gibbs
sampler.
"""
from typing import Dict

import torch

from gtbo.group_testing import bayes


def gibbs_kernel(
    particles: torch.Tensor,
    rho: float,
    log_posterior_params: Dict[str, torch.Tensor],
    log_base_measure_params: Dict[str, torch.Tensor],
    cycles: int = 2,
    liu_modification: bool = True,
    dtype: torch.dtype = torch.double,
    device: str = "cpu",
):
    """Applies a (Liu modified) Gibbs kernel (with MH) update.

    Implements vanilla (sequential, looping over coordinates) Gibbs sampling.
    When
    The Liu variant comes from Jun Liu's remarks in
    https://academic.oup.com/biomet/article-abstract/83/3/681/241540?redirectedFrom=fulltext

    which essentially changes the acceptance of a flip from
            p(flip) / [ p(no flip) + p(flip) ]
    to
            min(1, p(flip) / p(no flip) )

    In other words, Liu's modification increases the probability to flip.

    Args:
            particles: torch.Tensor [n_particles,n_patients] plausible infections states.
            rho: float, scaling for posterior.
            log_posterior_params: Dict of parameters to compute log-posterior.
            log_base_measure_params: Dict of parameters to compute log-base measure.
            cycles: the number of times we want to do Gibbs sampling.
            liu_modification : use or not Liu's modification.

    Returns:
            A np.array representing the new particles.
    """

    def gibbs_loop(
        i: int,
        particles: torch.Tensor,
        log_posteriors: torch.Tensor,
        dtype: torch.dtype = torch.double,
        device: str = "cpu",
    ):
        _i = i % num_patients
        # flip values at index i
        particles_flipped = particles  # .clone()
        particles_flipped[:, _i] = torch.logical_not(particles_flipped[:, _i])
        # compute log_posterior of flipped particles
        log_posteriors_flipped_at_i = bayes.tempered_logpos_logbase(
            particles_flipped,
            log_posterior_params,
            log_base_measure_params,
            rho,
            dtype=dtype,
            device=device,
        )
        # compute acceptance probability, depending on whether we use Liu mod.
        if liu_modification:
            log_proposal_ratio = log_posteriors_flipped_at_i - log_posteriors
        else:
            log_proposal_ratio = log_posteriors_flipped_at_i - torch.logaddexp(
                log_posteriors_flipped_at_i, log_posteriors
            )
        # here the MH thresholding is implicitly done.
        random_values = torch.rand(particles.shape[:1], dtype=dtype)
        flipped_at_i = torch.log(random_values) < log_proposal_ratio
        # selected_at_i = torch.logical_xor(flipped_at_i, particles[:, _i])
        selected_at_i = flipped_at_i == particles_flipped[:, _i]
        # particles = particles.clone()
        particles[:, _i] = selected_at_i
        log_posteriors = torch.where(
            flipped_at_i, log_posteriors_flipped_at_i, log_posteriors
        )
        return [particles, log_posteriors]

    num_patients = particles.shape[1]

    log_posteriors = bayes.tempered_logpos_logbase(
        particles,
        log_posterior_params,
        log_base_measure_params,
        rho,
        dtype=dtype,
        device=device,
    )

    for i in range(cycles * num_patients):
        particles, log_posteriors = gibbs_loop(
            i, particles, log_posteriors, dtype=dtype, device=device
        )

    # TODO(cuturi) : might be relevant to forward log_posterior_particles
    return particles


class Gibbs:
    """
    A Gibbs sampler.
    """

    def __init__(
        self,
        cycles: int = 2,
        liu_modification: bool = False,
        dtype: torch.dtype = torch.double,
        device: str = "cpu",
    ):
        """
        Initialize the Gibbs sampler.

        Args:
            cycles: the number of times we want to do Gibbs sampling.
            liu_modification: use or not Liu's modification.
            dtype: the type of the particles.
            device: the device to use.
        """
        self.cycles = cycles
        self.liu_modification = liu_modification
        self.model = None
        self.dtype = dtype
        self.device = device

    def __call__(
        self,
        particles: torch.Tensor,
        rho: float,
        log_posterior_params: Dict[str, torch.Tensor],
        log_base_measure_params: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Call the Gibbs sampler.

        Args:
            particles: the particles to update.
            rho: the temperature.
            log_posterior_params: the parameters of the log posterior.
            log_base_measure_params: the parameters of the log base measure.

        Returns:
            the updated particles.

        """
        return gibbs_kernel(
            particles,
            rho,
            log_posterior_params,
            log_base_measure_params,
            self.cycles,
            dtype=self.dtype,
            device=self.device,
        )
