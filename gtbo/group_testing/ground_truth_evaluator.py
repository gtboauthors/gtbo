from typing import List, Sequence, Tuple, Union, Optional

import gin
import numpy as np
import torch

from gtbo.benchmarks import Benchmark
from gtbo.util import from_unit_cube, to_unit_cube

dtype = torch.double


@gin.configurable
class GroundTruthEvaluator:
    """Evaluates points in the search space and replaces the wetlab in the original code."""

    def __init__(
        self,
        benchmark: Benchmark,
        default: torch.Tensor,
        n_default_samples: int,
        dtype: torch.dtype = torch.double,
        device: str = "cpu",
    ):
        """
        Initialize the ground truth evaluator.

        Args:
            benchmark: the benchmark to evaluate.
            default: the default configuration.
            n_default_samples: the number of samples to estimate the default.
            dtype: the dtype of the tensors.
            device: the device to run the computation on.
        """

        self.dim = int(benchmark.dim)
        self.benchmark = benchmark
        lbs = benchmark.lb_vec
        ubs = benchmark.ub_vec

        self.samples = torch.empty(size=(0, benchmark.dim), dtype=dtype)
        self.values = torch.empty(size=(0,), dtype=dtype)
        self.values_noiseless = torch.empty(size=(0,), dtype=dtype)
        self.n_default_samples = n_default_samples

        self.default = default
        self.downscaled_default = to_unit_cube(
            default, lb=benchmark.lb_vec, ub=benchmark.ub_vec
        )
        self.dtype = dtype
        self.device = device

        # Estimate default
        default_values_noisy_noiseless: List[
            Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]
        ] = [benchmark(self.default) for _ in range(n_default_samples)]
        if not benchmark.returns_noiseless:
            default_values_noisy_noiseless = [
                (dvn, dvn) for dvn in default_values_noisy_noiseless
            ]

        self.default_values = torch.tensor(
            [dvn[0].item() for dvn in default_values_noisy_noiseless],
            dtype=dtype,
        )
        self.default_values_noiseless = torch.tensor(
            [dvn[1].item() for dvn in default_values_noisy_noiseless],
            dtype=dtype,
        )

        self.default_mean = self.default_values.mean()
        self.default_std = self.default_values.std()
        self.samples = torch.cat((self.samples, default.repeat(n_default_samples, 1)))
        self.values = torch.cat(
            (self.values, self.default_values.to(dtype=torch.float64))
        )
        self.values_noiseless = torch.cat(
            (
                self.values_noiseless,
                self.default_values_noiseless.to(dtype=torch.float64),
            )
        )

        """
        Estimate noise and outputscale by splitting the search space into bins and take 
        sample variance among the bins that have the lowest (highest) absolute function value differences to the default.
        3*sqrt(D) bins are used, where D is the dimensionality of the search space. Sqrt(D) samples are used for the function
        variance and 2*sqrt(D) for the noise variance.            
        """
        perm = torch.randperm(self.dim)
        dims_per_bin = np.max([1, np.floor(np.sqrt(self.dim) / 3)]).astype(np.int64)
        directions = [
            perm[dims_per_bin * j : dims_per_bin * (j + 1)]
            for j in range(np.floor(self.dim / dims_per_bin).astype(np.int64))
        ]
        downscaled_direction_configs = self.downscaled_default.repeat(
            len(directions), 1
        )
        for j in range(len(directions)):
            downscaled_direction_configs[j, directions[j]] = 0.1 + 0.8 * torch.randint(
                0, 2, (directions[j].shape[0],)
            ).to(dtype=torch.double)
        direction_configs = from_unit_cube(downscaled_direction_configs, lbs, ubs)
        direction_values_noisy_noiseless = torch.tensor(
            [self.benchmark(config) for config in direction_configs]
        )
        if benchmark.returns_noiseless:
            direction_values = direction_values_noisy_noiseless[:, 0]
            direction_values_noiseless = direction_values_noisy_noiseless[:, 1]
        else:
            direction_values = direction_values_noisy_noiseless
            direction_values_noiseless = direction_values_noisy_noiseless
        direction_diffs = direction_values - self.default_mean
        signal_sample_limit_idx = min(
            benchmark.dim - 1,
            max(1, np.floor(2 * np.sqrt(benchmark.dim)).astype(np.int64)),
        )
        signal_sample_limit = (
            torch.abs(direction_diffs).sort().values[signal_sample_limit_idx]
        )
        noise_sample_limit_idx = min(
            benchmark.dim - 1,
            max(1, np.floor(np.sqrt(benchmark.dim)).astype(np.int64)),
        )
        noise_sample_limit = (
            torch.abs(direction_diffs).sort().values[noise_sample_limit_idx]
        )

        self.f_std = direction_diffs[
            torch.abs(direction_diffs) >= signal_sample_limit
        ].std()
        self.drift_std = direction_diffs[
            torch.abs(direction_diffs) <= noise_sample_limit
        ].std()
        self.pure_noise_std = self.default_std
        self.noise_std = torch.sqrt(self.drift_std**2 + self.pure_noise_std**2)

        self.samples = torch.cat((self.samples, direction_configs))
        self.values = torch.cat((self.values, direction_values))
        self.values_noiseless = torch.cat(
            (self.values_noiseless, direction_values_noiseless)
        )

    def __call__(
        self,
        active_dimensions: Sequence[int],
        tmp_default: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate a new configuration.

        Args:
            active_dimensions: the active dimensions of the function.
            tmp_default: the default configuration and its value.

        Returns:

        """

        if tmp_default is not None:
            default = tmp_default[0]
            downscaled_default = to_unit_cube(
                tmp_default[0], lb=self.benchmark.lb_vec, ub=self.benchmark.ub_vec
            )
            default_value = tmp_default[1]
        else:
            default = self.default
            downscaled_default = self.downscaled_default
            default_value = self.default_mean

        downscaled_other = torch.rand(self.benchmark.dim, dtype=dtype)
        for i in range(self.benchmark.dim):
            while torch.abs(downscaled_other[i] - downscaled_default[i]) < 0.40:
                downscaled_other[i] = torch.rand(size=(1,), dtype=dtype)
        other = from_unit_cube(
            downscaled_other, lb=self.benchmark.lb_vec, ub=self.benchmark.ub_vec
        )

        new_configuration = torch.tensor(
            [
                default[i].item() if i not in active_dimensions else other[i].item()
                for i in range(self.benchmark.dim)
            ],
            dtype=self.dtype,
        )

        y_new_noisy_noiseless = self.benchmark(new_configuration)

        if self.benchmark.returns_noiseless:
            y_new, y_new_noiseless = y_new_noisy_noiseless
        else:
            y_new, y_new_noiseless = y_new_noisy_noiseless, y_new_noisy_noiseless

        value = y_new
        value_noiseless = y_new_noiseless
        value_normalized = y_new - default_value
        # TODO: not good to extract the items above and build a new tensor here
        return (
            value_normalized,
            value,
            value_noiseless,
            new_configuration,
        )
