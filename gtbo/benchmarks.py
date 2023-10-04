import dataclasses
import json
import multiprocessing
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from logging import info
from multiprocessing import Pool
from pathlib import Path
from typing import List, Optional, Tuple, Type, Union

import gin
import numpy as np
import torch
from botorch.test_functions import (
    Branin as BotorchBranin,
    Griewank as BotorchGriewank,
    Hartmann as BotorchHartmann,
    Levy as BotorchLevy,
    SyntheticTestFunction,
)

from gtbo.util import eval_singularity_benchmark


@dataclass
class BenchmarkRequest:
    """
    A dataclass for a benchmark request for the BenchmarkSuite benchmarks.
    """

    function: str
    """The name of the problem. This has to be registered in the BenchmarkSuite"""
    dim: int
    """The dimensionality of the problem"""
    eval_points: List[List[float]]
    """The points to evaluate"""
    effective_dim: Optional[int] = None
    """The effective dimensionality of the problem"""
    noise_std: Optional[float] = None
    """The standard deviation of the noise"""
    max_steps: Optional[int] = 2000
    """The maximum number of steps to evaluate (this is irrelevant for benchmarks used in this work)"""

    def as_json(self) -> str:
        return json.dumps(dataclasses.asdict(self))


@gin.configurable
class Benchmark(ABC):
    """
    Abstract benchmark function.

    Args:
            dim: dimensionality of the objective function
            noise_std: the standard deviation of the noise (None means no noise)
            ub: the upper bound, the object will have the attribute ub_vec which is a np array of length dim filled with ub
            lb: the lower bound, the object will have the attribute lb_vec which is a np array of length dim filled with lb
    """

    def __init__(
        self,
        dim: int,
        ub: torch.Tensor,
        lb: torch.Tensor,
        noise_std: Optional[float],
        returns_noiseless: bool = False,
        **tkwargs,
    ):

        if (
            not lb.shape == ub.shape
            or not lb.ndim == 1
            or not ub.ndim == 1
            or not dim == len(lb) == len(ub)
        ):
            raise RuntimeError("bounds mismatch")
        if not torch.all(lb < ub):
            raise RuntimeError("out of bounds")
        self.noise_std = noise_std
        self._dim = dim

        dtype = tkwargs.get("dtype", torch.double)

        self._lb_vec = lb.to(dtype=dtype)
        self._ub_vec = ub.to(dtype=dtype)

        self.returns_noiseless = returns_noiseless

    @property
    def dim(self) -> int:
        """

        Returns:
                int: the benchmark dimensionality

        """
        return self._dim

    @property
    def lb_vec(self):
        """

        Returns:
                np.ndarray: the lower bound of the search space of this benchmark (length = benchmark dim)

        """
        return self._lb_vec

    @property
    def ub_vec(self):
        """

        Returns:
                np.ndarray: the upper bound of the search space of this benchmark (length = benchmark dim)

        """
        return self._ub_vec

    @property
    def fun_name(self) -> str:
        """

        Returns:
                str: the name of this function

        """
        return self.__class__.__name__

    def __call__(
        self, x: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Call the benchmark function for one or multiple points.
        Args:
            x: torch.Tensor: the x-value(s) to evaluate. numpy array can be 1 or 2-dimensional

        Returns: Tuple[torch.Tensor, torch.Tensor] the function values and the function values w/o noise

        """
        raise NotImplementedError()


class SyntheticBenchmark(Benchmark):
    """
    Abstract class for synthetic benchmarks

    Args:
            dim: the benchmark dimensionality
            ub: np.ndarray: the upper bound of the search space of this benchmark (length = benchmark dim)
            lb: np.ndarray: the lower bound of the search space of this benchmark (length = benchmark dim)
    """

    @abstractmethod
    def __init__(
        self,
        dim: int,
        ub: torch.Tensor,
        lb: torch.Tensor,
        noise_std: float,
        returns_noiseless: bool = False,
        *args,
        **tkwargs,
    ):
        super().__init__(
            dim=dim,
            ub=ub,
            lb=lb,
            noise_std=noise_std,
            returns_noiseless=returns_noiseless,
            **tkwargs,
        )

    @abstractmethod
    def __call__(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Call the benchmark function for one or multiple points.

        Args:
                x: Union[np.ndarray, List[float], List[List[float]]]: the x-value(s) to evaluate. numpy array can be 1 or 2-dimensional

        Returns:
                np.ndarray: The function values.


        """
        if x.ndim in [0, 1]:
            x = x.expand(1, -1)
        for y in x:
            if not torch.sum(y < self._lb_vec) == 0:
                raise RuntimeError("out of bounds")
            if not torch.sum(y > self._ub_vec) == 0:
                raise RuntimeError("out of bounds")

    @property
    def optimal_value(self) -> Optional[np.ndarray]:
        """

        Returns:
                Optional[Union[float, np.ndarray]]: the optimal value if known

        """
        return None


class EffectiveDimBenchmark(SyntheticBenchmark):
    """ """

    def __init__(
        self,
        dim: int,
        effective_dim: int,
        ub: np.ndarray,
        lb: np.ndarray,
        noise_std: float,
        returns_noiseless: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(
            dim=dim,
            ub=ub,
            lb=lb,
            noise_std=noise_std,
            returns_noiseless=returns_noiseless,
        )
        self.effective_dim: int = effective_dim

    @abstractmethod
    def __call__(
        self, x: Union[np.ndarray, List[float], List[List[float]]], *args, **kwargs
    ):
        raise NotImplementedError()


class BoTorchFunctionBenchmark(SyntheticBenchmark):
    def __init__(
        self,
        dim: int,
        noise_std: Optional[float],
        ub: torch.Tensor,
        lb: torch.Tensor,
        benchmark_func: Type[SyntheticTestFunction],
        *args,
        **tkwargs,
    ):
        super().__init__(
            dim=dim,
            ub=ub,
            lb=lb,
            noise_std=noise_std,
            **tkwargs,
        )
        try:
            self._benchmark_func = benchmark_func(dim=dim, noise_std=noise_std)
        except:
            self._benchmark_func = benchmark_func(noise_std=noise_std)

    @property
    def effective_dim(self) -> int:
        return self._dim

    @property
    def optimal_value(self) -> float:
        return self._benchmark_func.optimal_value

    def __call__(self, x, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        # check that x is in bounds
        assert torch.all(x <= self._ub_vec), f"x must be <= {self._ub_vec}, but is {x}"
        assert torch.all(x >= self._lb_vec), f"x must be >= {self._lb_vec}, but is {x}"

        super(self).__call__(x)
        if x.ndim in [0, 1]:
            x = x.expand(1, -1)
        assert x.ndim == 2
        res = self._benchmark_func.forward(
            torch.clip(x, self._lb_vec, self._ub_vec, noise=True)
        ).squeeze()
        res_noiseless = self._benchmark_func.forward(
            torch.clip(x, self._lb_vec, self._ub_vec, noise=False)
        ).squeeze()
        return res, res_noiseless


@gin.configurable
class EffectiveDimBoTorchBenchmark(BoTorchFunctionBenchmark):
    """
    A benchmark class for synthetic benchmarks with a known effective dimensionality that are based on a BoTorch
    implementation.

    Args:
            dim: int: the ambient dimensionality of the benchmark
            noise_std: float: standard deviation of the noise of the benchmark function
            effective_dim: int: the desired effective dimensionality of the benchmark function
            ub: np.ndarray: the upper bound of the benchmark search space. length = dim
            lb: np.ndarray: the lower bound of the benchmark search space. length = dim
            benchmark_func: Type[SyntheticTestFunction]: the BoTorch benchmark function to use
    """

    def __init__(
        self,
        dim: int,
        ub: torch.Tensor,
        lb: torch.Tensor,
        benchmark_func: Type[SyntheticTestFunction],
        effective_dim: int = 0,
        noise_std: float = 0.0,
        drift: float = 0.0,
        *args,
        **tkwargs,
    ):
        super().__init__(
            dim,
            noise_std,
            ub=ub,
            lb=lb,
            benchmark_func=benchmark_func,
            returns_noiseless=True,
            *args,
            **tkwargs,
        )
        if effective_dim > dim:
            raise RuntimeError("effective dim too large")
        assert effective_dim > 0, "effective dim must be > 0"
        self._fake_dim = dim
        self._effective_dim = effective_dim
        self.effective_dims = np.arange(dim)[:effective_dim]
        self.drift = drift
        info(f"effective dims: {list(self.effective_dims)}")

    def __call__(
        self, x, skip_checks: bool = False, *args, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if not skip_checks:
            # check that x is in bounds
            assert torch.all(
                x <= self._ub_vec
            ), f"x must be <= {self._ub_vec}, but is {x}"
            assert torch.all(
                x >= self._lb_vec
            ), f"x must be >= {self._lb_vec}, but is {x}"

        if x.ndim in [0, 1]:
            x = x.reshape(1, -1)
        assert x.ndim == 2
        res = self._benchmark_func.forward(
            torch.clip(x, self._lb_vec, self._ub_vec)[:, : self.effective_dim],
            noise=True,
        ).squeeze()
        res_noiseless = self._benchmark_func.forward(
            torch.clip(x, self._lb_vec, self._ub_vec)[:, : self.effective_dim],
            noise=False,
        ).squeeze()

        if self.drift:
            for i in range(1, int(np.floor(self.dim / self.effective_dim))):
                res += (
                    self.drift
                    * self._benchmark_func.forward(
                        torch.clip(x, self._lb_vec, self._ub_vec)[
                            :, i * self.effective_dim : (i + 1) * self.effective_dim
                        ],
                        noise=False,
                    ).squeeze()
                )
                res_noiseless += (
                    self.drift
                    * self._benchmark_func.forward(
                        torch.clip(x, self._lb_vec, self._ub_vec)[
                            :, i * self.effective_dim : (i + 1) * self.effective_dim
                        ],
                        noise=False,
                    ).squeeze()
                )
                break

        return res, res_noiseless

    @property
    def dim(self):
        return self._fake_dim

    @property
    def effective_dim(self) -> int:
        """

        Returns:
                int: the effective dimensionality of the benchmark

        """
        return self._effective_dim


@gin.configurable
class HartmannEffectiveDim(EffectiveDimBoTorchBenchmark):
    """
    A valley-shape benchmark function (see https://www.sfu.ca/~ssurjano/rosen.html)

    Args:
            dim: The ambient dimensionality of the function
            noise_std: The standard deviation of the noise
            effective_dim: The effective dimensionality of the function
    """

    def __init__(
        self,
        dim: int,
        *args,
        **kwargs,
    ):
        super().__init__(
            dim=dim,
            effective_dim=6,
            ub=torch.ones(dim),
            lb=torch.zeros(dim),
            benchmark_func=BotorchHartmann,
        )


@gin.configurable
class BraninEffectiveDim(EffectiveDimBoTorchBenchmark):
    """
    The Branin function with three local minima (see https://www.sfu.ca/~ssurjano/branin.html)

    Args:
            dim: The ambient dimensionality of the function
            noise_std: The standard deviation of the noise
            effective_dim: The effective dimensionality of the function
    """

    def __init__(
        self,
        dim: int,
        *args,
        **tkwargs,
    ):
        lb = torch.ones(dim) * (-5)
        lb[1] = 0
        ub = torch.ones(dim) * (15)
        ub[0] = 10

        super().__init__(
            dim=dim,
            effective_dim=2,
            lb=lb,
            ub=ub,
            benchmark_func=BotorchBranin,
            **tkwargs,
        )


@gin.configurable
class LevyEffectiveDim(EffectiveDimBoTorchBenchmark):
    """
    The Levy function with many local minima (see https://www.sfu.ca/~ssurjano/levy.html)

    Args:
            dim: The ambient dimensionality of the function
            noise_std: The standard deviation of the noise
            effective_dim: The effective dimensionality of the function
    """

    def __init__(self, dim: int, *args, **kwargs):
        super(LevyEffectiveDim, self).__init__(
            dim=dim,
            lb=torch.ones(dim) * (-10),
            ub=torch.ones(dim) * 10,
            benchmark_func=BotorchLevy,
        )


@gin.configurable
class GriewankEffectiveDim(EffectiveDimBoTorchBenchmark):
    """
    The Griewank function with many local minima (see https://www.sfu.ca/~ssurjano/griewank.html)

    .. warning::
        This function has its optimum at the origin. This might give a misleading performance for GTBO if
        the default is at the origin. This is why we change the default for this benchmark.


    Args:
            dim: The ambient dimensionality of the function
            noise_std: The standard deviation of the noise
            effective_dim: The effective dimensionality of the function
    """

    def __init__(self, dim: int, *args, **kwargs):
        super(GriewankEffectiveDim, self).__init__(
            dim=dim,
            lb=torch.ones(dim) * (-600),
            ub=torch.ones(dim) * 600,
            benchmark_func=BotorchGriewank,
        )
        self.default = torch.ones(dim) * 100.0


@gin.configurable
class SingularityBenchmark(Benchmark):
    """
    Abstract class for a benchmark function that is evaluated using the BenchmarkSuite singularity image.
    """

    def __init__(
        self,
        dim: int,
        name,
        singularity_image_path: Optional[str] = None,
        n_workers: Optional[int] = None,
    ):
        """
        Initialize the benchmark.

        Args:
            dim: the dimensionality of the benchmark
            name: the name of the benchmark (this has to be the name of the benchmark in the BenchmarkSuite singularity image)
            singularity_image_path: the path to the BenchmarkSuite singularity image
            n_workers: the number of workers to use for parallel evaluation
        """
        super().__init__(
            dim=dim, ub=torch.ones(dim), lb=torch.zeros(dim), noise_std=None
        )

        if singularity_image_path is None:
            singularity_image_path = os.path.join(
                Path(__file__).parent.parent.parent.resolve(), "BenchSuite"
            )

        self.singularity_image_path = singularity_image_path
        self.name = name
        self.n_workers = n_workers

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # check that x is in bounds
        assert torch.all(x <= self._ub_vec), f"x must be <= {self._ub_vec}, but is {x}"
        assert torch.all(x >= self._lb_vec), f"x must be >= {self._lb_vec}, but is {x}"

        if x.ndim == 1:
            x = x.unsqueeze(0)
        with Pool(
            multiprocessing.cpu_count() if self.n_workers is None else self.n_workers
        ) as p:
            func = partial(
                eval_singularity_benchmark,
                singularity_image_path=self.singularity_image_path,
                name=self.name,
            )
            results = p.map(func, x.detach().cpu().numpy().tolist())
        # TODO make dtype dynamic
        results = torch.tensor(results, dtype=torch.float64)
        return results


@gin.configurable
class LassoDNA(SingularityBenchmark):
    """
    The LassoDNA benchmark function (see https://github.com/ksehic/LassoBench)

    This benchmark has 180 dimensions and is evaluated using the BenchmarkSuite singularity image.
    """

    def __init__(
        self,
    ):
        dim = 180
        super().__init__(
            dim=dim,
            name="lasso_dna",
            n_workers=1,
        )


@gin.configurable
class Mopta08(SingularityBenchmark):
    """
    The Mopta08 benchmark function (see https://www.pseven.io/blog/use-cases/pseven-beats-mopta08-automotive-benchmark.html)

    This benchmark has 124 dimensions and is evaluated using the BenchmarkSuite singularity image.
    """

    def __init__(
        self,
    ):
        dim = 124
        super().__init__(
            dim=dim,
            name="mopta08",
        )
