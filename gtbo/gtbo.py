import copy
import json
import logging
import lzma
import os
import time
import warnings
import zlib
from argparse import ArgumentError
from datetime import datetime
from typing import Optional, List, Any

import gin
import numpy as np
import torch
from botorch.acquisition import qLogNoisyExpectedImprovement
from botorch.sampling import SobolQMCNormalSampler
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.kernels import AdditiveKernel
from torch import Tensor
from torch.quasirandom import SobolEngine

from gtbo.benchmarks import Benchmark
from gtbo.candidates import create_candidates
from gtbo.gaussian_process import (
    fit_mll,
    get_gp,
    robust_optimize_acqf,
    test_mlls_and_return_best_state_dict,
)
from gtbo.group_testing.ground_truth_evaluator import GroundTruthEvaluator
from gtbo.group_testing.tester import GroupTester
from gtbo.group_testing.utils import BColors
from gtbo.util import from_unit_cube, to_unit_cube


@gin.configurable
class GTBO:
    """
    Implementation of the GTBO algorithm.
    """

    def __init__(
        self,
        benchmark: Benchmark,
        maximum_number_evaluations: int,
        number_initial_points: int,
        results_dir: str,
        device="cpu",
        dtype: str = "float64",
        logging_level: str = "info",
        retrain_gp_from_scratch_every: int = 50,
    ):
        """
        Initialize the GTBO algorithm.

        Args:
            benchmark: The benchmark to use
            maximum_number_evaluations: the maximum number of evaluations
            number_initial_points: the number of initial points to sample
            results_dir: the directory to save the results to
            device: the device to use
            dtype: the dtype to use
            logging_level: the logging level
            retrain_gp_from_scratch_every: the number of iterations after which to retrain the GP from scratch
        """
        self.benchmark = benchmark
        """The benchmark object"""
        self.benchmark.device = "cpu"
        """The device to use"""
        self.logging_level = logging_level
        """The logging level"""
        self.maximum_number_evaluations = maximum_number_evaluations
        """The maximum number of evaluations"""
        self.number_initial_points = number_initial_points
        """The number of initial points to sample"""
        self.retrain_gp_from_scratch_every = retrain_gp_from_scratch_every
        """The number of iterations after which to retrain the GP from scratch"""
        self._n_evals = 0
        self.device = device if torch.cuda.is_available() else "cpu"
        """The device to use"""
        logging.info(f"Using device {self.device}")
        self.dtype = None
        """The dtype to use"""
        if dtype == "float64":
            self.dtype = torch.float64
        elif dtype == "float32":
            self.dtype = torch.float32
        else:
            raise ValueError(f"Unknown dtype {dtype}")

        if self.number_initial_points <= 0:
            raise ArgumentError(
                f"Number of initial points must be positive but is {self.number_initial_points}"
            )
        if self.benchmark.dim < 6:
            raise ArgumentError(
                f"Dimensionality must be at least 6 but is {self.benchmark.dim}. This is due to the binning strategy and rounding issues."
            )

        now = datetime.now()
        gin_config_str = gin.operative_config_str()
        adler = zlib.adler32(gin_config_str.encode("utf-8"))

        fname = now.strftime("%m-%d-%Y-%H:%M:%S:%f")
        self.results_dir = os.path.join(results_dir, str(adler), fname)
        """The directory to save the results to"""
        os.makedirs(self.results_dir, exist_ok=True)
        # save gin config to file
        with open(os.path.join(self.results_dir, "gin_config.txt"), "w") as f:
            f.write(gin.operative_config_str())

    def run(self) -> None:
        """
        Run the GTBO algorithm. This consists of two phases: the group testing phase and the Bayesian optimization phase.
        In the group testing phase, we identify the active dimensions of the problem.
        In the Bayesian optimization phase, we optimize the objective function using the identified active dimensions
        by placing a shorter lengthscale on them.

        Returns:
            None.

        """
        try:
            benchmark_default = self.benchmark.default
        except:
            benchmark_default = (
                self.benchmark.ub_vec - self.benchmark.lb_vec
            ) / 2 + self.benchmark.lb_vec

        ### GT PHASE ###

        # Run the ground truth phase if we are not continuing from a previous run and we are not running BO only
        ground_truth_evaluator = GroundTruthEvaluator(
            benchmark=self.benchmark,
            default=benchmark_default,
            dtype=self.dtype,
            device=self.device,
        )

        signal_std = torch.sqrt(
            ground_truth_evaluator.f_std**2
            + ground_truth_evaluator.pure_noise_std**2
        ).item()
        noise_std = ground_truth_evaluator.noise_std.item()
        if noise_std < 1e-6:
            noise_std = 1e-6

        group_tester = GroupTester(
            evaluator=ground_truth_evaluator,
            noise_std=noise_std,
            signal_std=signal_std,
            device=self.device,
            dtype=self.dtype,
            results_dir=self.results_dir,
        )

        with lzma.open(os.path.join(self.results_dir, "sigma_signal.txt.xz"), "w") as f:
            f.write(str(group_tester.signal_std).encode("utf-8"))
        with lzma.open(os.path.join(self.results_dir, "sigma_noise.txt.xz"), "w") as f:
            f.write(str(group_tester.noise_std).encode("utf-8"))
        with lzma.open(
            os.path.join(self.results_dir, "sigma_pure_noise.txt.xz"), "w"
        ) as f:
            f.write(str(ground_truth_evaluator.pure_noise_std).encode("utf-8"))

        state = group_tester.optimize()

        test_times = dict(
            tester_optimize_time=group_tester.optimize_time,
            tester_gamma_time=group_tester.gamma_time,
            tester_entropy_time=group_tester.entropy_time,
            tester_conditional_entropy_time=group_tester.conditional_entropy_time,
            tester_eval_time=group_tester.eval_time,
            tester_produce_sample_time=group_tester.produce_sample_time,
            tester_update_time=group_tester.resample_time,
            tester_resample_time=group_tester.sampler.resample_time,
            tester_move_time=group_tester.sampler.move_time,
            tester_resample_move_time=group_tester.sampler.resample_move_time,
            tester_sample_times=group_tester.sample_times,
            bo_times=[],
            gp_fit_times=[],
            acq_opt_times=[],
        )
        marginals_np = state.marginals[0].cpu().numpy()

        # those are unnormalized
        gt_x = group_tester.x
        gt_fx = group_tester.fx
        gt_fx_noiseless = group_tester.fx_noiseless

        n_active = np.sum(marginals_np > group_tester.activeness_threshold)
        active_dims = np.where(marginals_np > group_tester.activeness_threshold)[0]

        gt_subset_indices = None
        _, gt_subset_indices = np.unique(
            np.sign((gt_x - benchmark_default)[:, active_dims]),
            axis=0,
            return_index=True,
        )

        if gt_subset_indices is not None:
            gt_x = gt_x[gt_subset_indices]
            gt_fx = gt_fx[gt_subset_indices]
            gt_fx_noiseless = gt_fx_noiseless[gt_subset_indices]

        logging.info(
            f"{BColors.BOLD}Active dimensions: {n_active}/{self.benchmark.dim}{BColors.ENDC}"
        )
        self._n_evals = gt_x.shape[0]

        bo_time_start = time.time()

        target_dim = self.benchmark.dim

        #### SETUP FOR BO ####

        # reuse GT samples
        x = (
            to_unit_cube(
                gt_x,
                lb=self.benchmark.lb_vec,
                ub=self.benchmark.ub_vec,
            )
            .clone()
            .detach()
            .cpu()
        )
        fx = gt_fx.clone().detach().cpu()
        fx_noiseless = gt_fx_noiseless.clone().detach().cpu()
        remove_first_samples = ground_truth_evaluator.n_default_samples

        n_init_samples = min(
            max(self.maximum_number_evaluations - x.shape[0], 1),
            self.number_initial_points,
        )
        x_init = (
            SobolEngine(target_dim, scramble=True)
            .draw(n_init_samples)
            .to(dtype=self.dtype)
        )

        x_init_up = from_unit_cube(
            x=x_init,
            lb=self.benchmark.lb_vec,
            ub=self.benchmark.ub_vec,
        )

        fx_init_noisy_noiseless = self.benchmark(x_init_up)
        if self.benchmark.returns_noiseless:
            fx_init, fx_init_noiseless = fx_init_noisy_noiseless
        else:
            fx_init = torch.clone(fx_init_noisy_noiseless)
            fx_init_noiseless = torch.clone(fx_init_noisy_noiseless)
        fx = torch.cat((fx, fx_init.clone().detach().cpu().reshape(-1)))
        fx_noiseless = torch.cat(
            (fx_noiseless, fx_init_noiseless.clone().detach().cpu().reshape(-1))
        )
        self._n_evals += x_init.shape[0]
        x = torch.cat((x, x_init), dim=0)

        # NOTE THAT AT THIS POINT gt_x is in original scale whereas x is in [0,1]

        if x.shape[0] == 0:
            warnings.warn("No initial points were sampled.")
        hyperparameters = torch.empty(0)
        # model_hyperparameters is the gp statedict that we use to reinitialize the gp with the same hps
        model_hyperparameters = None
        prev_mll = None
        while self._n_evals <= self.maximum_number_evaluations:

            if self._n_evals % self.retrain_gp_from_scratch_every == 0:
                model_hyperparameters = None

            bo_iter_start_time = time.time()
            # TODO hp_summary and model_hyperparameters contain the same data so we only need one of them
            (
                x_best,
                loss,
                hp_summary,
                model_hyperparameters,
                gp_fit_time,
                acq_opt_time,
                prev_mll,
            ) = create_candidates(
                x=x,
                fx=fx,
                device=self.device,
                remove_first_samples=remove_first_samples,
                active_dimensions=active_dims,
                model_hyperparameters=model_hyperparameters,
                prev_mll=prev_mll,
            )

            logging.debug(f"GP fitting took {gp_fit_time:.3f}s")

            hyperparameters = torch.cat((hyperparameters, hp_summary.to(device="cpu")))
            x_next = x_best[loss.argmin()].detach().cpu().reshape(1, -1)
            x_next_up = from_unit_cube(
                x_next,
                self.benchmark.lb_vec,
                self.benchmark.ub_vec,
            )

            y_next_noisy_noiseless = self.benchmark(x_next_up)
            if self.benchmark.returns_noiseless:
                y_next, y_next_noiseless = y_next_noisy_noiseless
            else:
                y_next, y_next_noiseless = torch.clone(
                    y_next_noisy_noiseless
                ), torch.clone(y_next_noisy_noiseless)
            y_next = y_next.squeeze()
            if y_next < fx.min():
                logging.info(
                    f"{BColors.OKBLUE}({self._n_evals}){BColors.ENDC} -> {BColors.BOLD}New best found: {BColors.OKGREEN}{y_next:.3f}{BColors.ENDC}"
                )
            else:
                logging.info(
                    f"{BColors.OKBLUE}({self._n_evals}){BColors.ENDC} -> New best not found: {y_next:.3f}, best is {fx.min():.3f}, noiseless best: {fx_noiseless.min():.5f}"
                )

            x = torch.cat((x, x_next))
            fx = torch.cat((fx, y_next.reshape(-1)))
            fx_noiseless = torch.cat((fx_noiseless, y_next_noiseless.reshape(-1)))

            self._n_evals += 1

            test_times["bo_times"].append(time.time() - bo_iter_start_time)
            test_times["gp_fit_times"].append(gp_fit_time)
            test_times["acq_opt_times"].append(acq_opt_time)

            if (
                self._n_evals % 10 == 0
                or self._n_evals > self.maximum_number_evaluations
            ):
                with open(os.path.join(self.results_dir, "test_times.json"), "w") as f:
                    json.dump(test_times, f)

                with lzma.open(
                    os.path.join(self.results_dir, "results.csv.xz"), "w"
                ) as f:
                    x_up = from_unit_cube(
                        x, self.benchmark.lb_vec, self.benchmark.ub_vec
                    )
                    np.savetxt(
                        f,
                        np.hstack(
                            [
                                x_up.detach().cpu().numpy(),
                                fx.detach().cpu().numpy().reshape(-1, 1),
                            ]
                        ),
                        delimiter=",",
                    )

                # save noiseless function values to disk
                with lzma.open(
                    os.path.join(self.results_dir, "results_noiseless.csv.xz"), "w"
                ) as f:
                    np.savetxt(
                        f,
                        np.hstack(
                            [
                                x_up.detach().cpu().numpy(),
                                fx_noiseless.detach().cpu().numpy().reshape(-1, 1),
                            ]
                        ),
                        delimiter=",",
                    )

                with lzma.open(os.path.join(self.results_dir, "hps.csv.xz"), "w") as f:
                    np.savetxt(f, hyperparameters.detach().cpu().numpy(), fmt="%f")

            try:
                with open(os.path.join(self.results_dir, "test_times.json"), "w") as f:
                    json.dump(test_times, f)
            except Exception:
                logging.warning(f"Failed to save test times")

        bo_time_end = time.time()

        test_times["bo_time"] = bo_time_end - bo_time_start
        with open(os.path.join(self.results_dir, "test_times.json"), "w") as f:
            json.dump(test_times, f)
