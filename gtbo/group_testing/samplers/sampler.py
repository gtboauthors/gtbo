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
"""Base class for particles samplers."""
import torch


class Sampler:
    """
    Base class for (particle) samplers.
    """

    def __init__(self):
        self.particle_weights = None
        self.particles = None
        self.convergence_metric = torch.nan

    def reset(self) -> None:
        """
        Resets the sampler to its initial state.

        Returns:
            None.

        """
        self.particle_weights = None
        self.particles = None
        self.convergence_metric = torch.nan

    @property
    def marginal(self) -> torch.Tensor:
        """
        Returns the marginal distribution of the posterior currently stored.

        Returns:
            torch.Tensor. The marginal distribution of the posterior currently stored.

        """
        return torch.sum(self.particle_weights.reshape((-1, 1)) * self.particles, dim=0)

    def reset_convergence_metric(self) -> None:
        """
        Resets the convergence metric.

        Returns:
            None.

        """
        self.convergence_metric = torch.nan
