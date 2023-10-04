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
"""Sequential Monte-Carlo Sampler in jax."""
import logging
import time
from random import random
from typing import Dict

import numpy as np
import torch

from gtbo.group_testing import bayes
from gtbo.group_testing.samplers import kernels, sampler, temperature
from gtbo.group_testing.state import State


class SmcSampler(sampler.Sampler):
    """Sequential monte carlo sampler."""

    NAME = "smc"

    def __init__(
        self,
        kernel=kernels.Gibbs(),
        resample_at_each_iteration: bool = False,
        start_from_prior: bool = True,
        num_particles: int = 10000,
        min_kernel_iterations: int = 2,
        max_kernel_iterations: int = 20,
        min_ratio_delta: float = 0.02,
        target_unique_ratio: float = 0.95,
        dtype: torch.dtype = torch.double,
        device: str = "cpu",
    ):
        """Initializes SmcSampler object.

        Args:
            kernel: function tasked with perturbing/refreshing particles
            resample_at_each_iteration: bool, in sequential setting, boolean that indicates whether particles should be resampled from scratch when adding new test results (True), or whether previous particles should be used as a starting point to recover particle approximations for new posterior
            start_from_prior: if True, initial batch of particles is sampled from prior distribution. if False, we use uniform sampling on {0,1}^num_patients.
            num_particles: number of particles used in Smc approximation
            min_kernel_iterations: minimal number of times MH kernel is applied
            max_kernel_iterations: maximal number of times MH kernel is applied
            min_ratio_delta: when difference (delta) between two consecutive unique ratio values goes below that value, MH kernel refreshes is stopped.
            target_unique_ratio: when unique ratio (number of unique particles / num_particles) goes above that value we stop.
        """
        super().__init__()
        self._kernel = kernel
        kernel.device = device
        kernel.dtype = dtype
        self._resample_at_each_iteration = resample_at_each_iteration
        self._start_from_prior = start_from_prior
        self._num_particles = num_particles
        self._min_kernel_iterations = min_kernel_iterations
        self._max_kernel_iterations = max_kernel_iterations
        self._min_ratio_delta = min_ratio_delta
        self._target_unique_ratio = target_unique_ratio
        self._sampled_up_to = 0
        self.resample_time = 0
        self.move_time = 0
        self.resample_move_time = 0

        self.dtype = dtype
        self.device = device

    def reset(self):
        super().reset()
        self._sampled_up_to = 0

    def produce_sample(self, state: State):
        """Produces a particle approx to posterior distribution given tests.

        If no tests have been carried out so far, naively sample from
        prior distribution.

        Otherwise take into account previous tests to form posterior
        and sample from it using a SMC sampler.

        Args:
            state: the current state of what has been tested, etc.

        Returns:
            a measure of the quality of convergence, here the ESS
            also updates particle_weights and particles members.
        """

        shape = (self._num_particles, state.num_patients)
        if len(state.past_test_results) == 0:
            self.particles = torch.rand(shape) < state.prior_infection_rate
            self.particle_weights = (
                torch.ones((self._num_particles,), dtype=self.dtype)
                / self._num_particles
            )
        else:
            # if we have never sampled before particles, resample field is True
            sampling_from_scratch = (
                self._resample_at_each_iteration or self.particles is None
            )
            # if we resample, either sample uniformly on {0,1}^num_patients, or prior
            if sampling_from_scratch:
                # if we start from prior, use prior_infection_rate otherwise uniform
                threshold = (
                    state.prior_infection_rate if self._start_from_prior else 0.5
                )
                particles = torch.rand(shape) < threshold
            # else, we recover the latest particles that were sampled previously
            else:
                particles = self.particles
            # sample now from posterior
            trm = time.time()
            self.particle_weights, self.particles = self.resample_move(
                particles, state, sampling_from_scratch
            )
            self.resample_move_time += time.time() - trm
            self._sampled_up_to = state.past_test_results.shape[0]
        # keeping track of ESS as convergence metric for SMC sampler
        self.convergence_metric = temperature.effective_sample_size(
            1, torch.log(self.particle_weights)
        )

    def resample_move(self, particles, state: State, sampling_from_scratch):
        """Resample / Move sequence."""
        # recover log_posterior params from state. if we resample, only recover
        # latest wave of tests. if not resampling, get entire information of prior
        #
        log_posterior_params = state.get_log_posterior_params(
            sampling_from_scratch, self._start_from_prior, self._sampled_up_to
        )
        log_base_measure_params = state.get_log_base_measure_params(
            sampling_from_scratch, self._start_from_prior, self._sampled_up_to
        )
        # add log weights to dictionary of log_prior parameters
        log_posterior = bayes.log_probability(
            particles, device=self.device, dtype=self.dtype, **log_posterior_params
        )
        alpha, log_tempered_probability = temperature.find_step_length(0, log_posterior)
        rho = alpha
        particle_weights = temperature.importance_weights(log_tempered_probability)
        logging.debug(f"Sampling {rho:.2%}")
        while rho < 1:
            logging.debug(f"Sampling {rho:.2%}")
            tr = time.time()
            particles = particles[self.resample(particle_weights), :]
            self.resample_time += time.time() - tr
            tm = time.time()
            particles = self.move(
                particles, rho, log_posterior_params, log_base_measure_params
            )
            self.move_time += time.time() - tm
            log_posterior = bayes.log_probability(
                particles, device=self.device, dtype=self.dtype, **log_posterior_params
            )
            alpha, log_tempered_probability = temperature.find_step_length(
                rho, log_posterior
            )
            particle_weights = temperature.importance_weights(log_tempered_probability)
            rho = rho + alpha
        return particle_weights, particles

    def resample(
        self,
        particle_weights: torch.Tensor,
    ) -> list:
        """Systematic resampling given weights.

        This is coded in onp because it seems to be faster than lax here.

        Args:
            particle_weights: the weights of the particles.

        Returns:
            Return a list of indices.
        """
        num_particles = particle_weights.shape[0]
        noise = random()
        positions = (noise + np.arange(num_particles)) / num_particles
        cum_sum = np.cumsum(particle_weights.cpu().numpy(), axis=0)
        cum_sum[-1] = 1  # fix to handle numerical issues where sum isn't exactly 1
        indices = list()
        i, j = 0, 0
        while i < num_particles:
            if positions[i] < cum_sum[j]:
                indices.append(j)
                i += 1
            else:
                j += 1
        return indices

    def move(
        self,
        particles: torch.Tensor,
        rho: float,
        log_posterior_params: Dict[str, torch.Tensor],
        log_base_measure_params: Dict[str, torch.Tensor],
    ):
        """Applies the kernel until particles are sufficiently refreshed.

        Args:
            particles : current particles
            rho : temperature, between 0 and 1
            log_posterior_params : dict of parameters to evaluate posterior
            log_base_measure_params : dict of parameters to evaluate base measure

        Returns:
            a torch.Tensor of particles of the same size as the input.
        """
        num_particles = particles.shape[0]
        unique_ratio = torch.tensor(
            np.unique(particles.numpy(), axis=0).shape[0] / num_particles
        )

        for it in range(self._max_kernel_iterations):
            particles = self._kernel(
                particles, rho, log_posterior_params, log_base_measure_params
            )

            old_unique_ratio = unique_ratio
            unique_ratio = torch.tensor(
                np.unique(particles.numpy(), axis=0).shape[0] / num_particles
            )
            if it < self._min_kernel_iterations - 1:
                continue
            ratio_delta = torch.abs(old_unique_ratio - unique_ratio)
            if (
                ratio_delta <= self._min_ratio_delta
                or unique_ratio > self._target_unique_ratio
            ):
                break

        return particles
