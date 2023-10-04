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
"""A state of the group testing algorithm."""

from typing import Dict

import torch

from gtbo.group_testing.samplers.sampler import Sampler


class State:
    """
    A state of the group testing algorithm.

    Attributes:
            device: str. The device on which the tensors are stored.
            dtype: torch.dtype. The dtype of the tensors.
            signal_variance: float. The variance of the signal.
            noise_variance: float. The variance of the noise.
            num_patients: int. The number of patients.
            num_tests_per_cycle: int. The number of tests per cycle.
            prior_infection_rate: float. The prior infection rate.
    """

    def __init__(
        self,
        num_patients: int,
        num_tests_per_cycle: int,
        prior_infection_rate: float,
        noise_variance: float,
        signal_variance: float,
        device: str = "cpu",
        dtype: torch.dtype = torch.double,
    ):
        self.device = device
        self.dtype = dtype

        self.signal_variance = signal_variance
        self.noise_variance = noise_variance
        self.num_patients = num_patients
        self.num_tests_per_cycle = num_tests_per_cycle

        self.prior_infection_rate = prior_infection_rate

        self.curr_cycle = 0
        self.past_groups = None
        self.past_test_results = None
        self.groups_to_test = None
        self.particle_weights = None
        self.particles = None
        self._particles_long = None
        self.to_clear_positives = None
        self.all_cleared = False
        self.marginals = {}
        self.reset()  # Initializes the attributes above.

    def reset(self):
        """Reset the state."""
        self.curr_cycle = 0
        self.past_groups = torch.empty(size=(0, self.num_patients), dtype=torch.bool)
        self.past_test_results = torch.empty(size=(0,), dtype=torch.bool)
        self.groups_to_test = torch.empty(size=(0, self.num_patients), dtype=torch.bool)

        # Those are specific to some methods. They are not always used or filled.
        self.particle_weights = None
        self.particles = None
        self._particles_long = None
        self.to_clear_positives = torch.empty(size=(0,), dtype=torch.bool)
        self.all_cleared = False

        # In case we store marginals computed in different ways.
        self.marginals = {}

    def add_test_results(self, test_results: torch.Tensor) -> None:
        """
        Update state with results from recently tested groups.

        Args:
            test_results: torch.Tensor. The test results.

        Returns:
            None.

        """
        """Update state with results from recently tested groups."""
        self.past_test_results = torch.concatenate(
            (self.past_test_results, test_results), axis=0
        )

        missing_entries_in_to_clear = len(self.past_test_results) - len(
            self.to_clear_positives
        )
        if missing_entries_in_to_clear > 0:
            # we should update the list of groups that have been tested positives.
            # this information is used by some strategies, notably Dorfman type ones.
            # if some entries are missing, they come by default from the latest wave
            # of tests carried out.
            self.to_clear_positives = torch.concatenate(
                (self.to_clear_positives, test_results[-missing_entries_in_to_clear:]),
                axis=0,
            )

    def add_past_groups(self, groups: torch.Tensor) -> None:
        """
        Update state with groups tested in the past.

        Args:
            groups: torch.Tensor. The groups tested in the past.

        Returns:
            None.

        """
        self.past_groups = torch.concatenate((self.past_groups, groups), axis=0)

    def update_particles(self, sampler: Sampler) -> None:
        """
        Keep as current particles the ones of a particle sampler.

        Args:
            sampler: Sampler. The sampler.

        Returns:
            None.

        """
        self.particle_weights = sampler.particle_weights
        self.particles = sampler.particles
        self._particles_long = self.particles.cpu().to(torch.int64)

    def get_log_posterior_params(
        self, sampling_from_scratch=True, start_from_prior=False, sampled_up_to=0
    ) -> Dict[str, torch.Tensor]:
        """Outputs parameters used to compute log posterior.

        Two scenarios are possible, depending on whether one wants to update an
        existing posterior approximation, or whether one wants to resample it from
        scratch
        Args:
          sampling_from_scratch: bool, flag to select all tests / prior seen so far
                or only the last wave of tests results.
          start_from_prior: bool, flag to indicate whether the first particles have
                been sampled from prior (True) or from a uniform measure.
          sampled_up_to: indicates what tests were used previously to generate
                samples. used when sampling_from_scratch is False


        Returns:
          a dict structure with fields relevant to evaluate Bayes.log_posterior
        """
        log_posterior_params = dict(
            noise_variance=self.noise_variance,
            signal_variance=self.signal_variance,
        )
        # if past groups are same as latest, use by all past, including prior.
        if sampling_from_scratch or sampled_up_to == 0:
            log_posterior_params.update(
                test_results=self.past_test_results, groups=self.past_groups
            )
            if start_from_prior:
                log_posterior_params.update(prior_infection_rate=None)
            else:
                log_posterior_params.update(
                    prior_infection_rate=self.prior_infection_rate
                )
        # if only using latest wave of tests, use no prior in posterior and only
        # use tests added since sampler was last asked to produce_sample.
        else:
            log_posterior_params.update(
                test_results=self.past_test_results[sampled_up_to:],
                groups=self.past_groups[sampled_up_to:],
                prior_infection_rate=None,
            )
        return log_posterior_params

    def get_log_base_measure_params(
        self, sampling_from_scratch=True, start_from_prior=False, sampled_up_to=0
    ) -> Dict[str, torch.Tensor]:
        """Outputs parameters used to compute log probability of base measure.

        Two scenarios are possible, depending on whether one wants to update an
        existing posterior approximation, or whether one wants to resample it from
        scratch
        Args:
          sampling_from_scratch: bool, flag to select all tests / prior seen so far
                or only the last wave of tests results.
          start_from_prior: bool, flag to indicate whether the first particles have
                been sampled from prior (True) or from a uniform measure.
          sampled_up_to: indicates what tests were used previously to generate
                samples. used when sampling_from_scratch is False

        Returns:
          a dict structure with fields relevant to evaluate Bayes.log_posterior
        """
        log_base_measure_params = dict(
            noise_variance=self.noise_variance,
            signal_variance=self.signal_variance,
        )
        if sampling_from_scratch or sampled_up_to == 0:
            log_base_measure_params.update(test_results=None, groups=None)
            if start_from_prior:
                log_base_measure_params.update(
                    prior_infection_rate=self.prior_infection_rate
                )
            else:
                log_base_measure_params.update(prior_infection_rate=None)

        else:
            past_minus_unused_tests = self.past_test_results[:sampled_up_to]
            past_minus_unused_groups = self.past_groups[:sampled_up_to]
            log_base_measure_params.update(
                test_results=past_minus_unused_tests,
                groups=past_minus_unused_groups,
                prior_infection_rate=self.prior_infection_rate,
            )
        return log_base_measure_params
