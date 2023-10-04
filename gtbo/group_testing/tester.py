import logging
import lzma
import math
import os.path
import time
from typing import Tuple

import gin
import numpy as np
import torch
from torch.distributions import Normal

from gtbo.group_testing.ground_truth_evaluator import GroundTruthEvaluator
from gtbo.group_testing.samplers.sequential_monte_carlo import SmcSampler
from gtbo.group_testing.state import State
from gtbo.group_testing.utils import BColors


@gin.configurable
class GroupTester:
    """
    The main class of the group testing algorithm.
    """

    def __init__(
        self,
        evaluator: GroundTruthEvaluator,
        noise_std: float,
        signal_std: float,
        gt_iterations: int = 100,
        n_particles: int = 5000,
        n_normal_samples: int = 1000,
        n_initial_groups: int = 3,
        prior_activeness_rate: float = 0.05,
        dtype: torch.dtype = torch.double,
        lower_convergence_threshold: float = 0.01,
        upper_convergence_threshold: float = 0.9,
        test_only_non_converged_lb: float = 0.01,
        test_only_non_converged_ub: float = 0.9,
        activeness_threshold: float = 0.5,
        device: str = "cpu",
        results_dir: str = None,
        max_groups_per_iteration: int = 1,
        maximum_mutual_information_loss: float = 0.1,
    ):
        """
        Initializes the GroupTester object.

        Args:
            evaluator: a GroundTruthEvaluator object.
            noise_std: the standard deviation of the noise when evaluating the same point multiple times
            signal_std: the standard deviation of the signal when evaluating random points
            gt_iterations: the number of iterations of the group testing algorithm
            n_particles: the number of particles used in the SMC sampler
            n_normal_samples: the number of samples used to estimate the entropy
            n_initial_groups: the number of initial groups to test
            prior_activeness_rate: the prior probability of a patient being active
            dtype: the dtype of the torch tensors
            lower_convergence_threshold: marginals below this are considered converged
            upper_convergence_threshold: marginals above this are considered converged
            test_only_non_converged_lb: only test groups with marginals above this
            test_only_non_converged_ub: only test groups with marginals below this
            activeness_threshold: the threshold for a patient to be considered active
            device: the device to run the computation on
            results_dir: the directory to save the results to
            max_groups_per_iteration: the maximum number of groups to test per iteration
            maximum_mutual_information_loss: the maximum loss in mutual information to allow when selecting groups
        """
        self.evaluator = evaluator
        self.gt_iterations = gt_iterations
        self.sampler = SmcSampler(
            num_particles=n_particles,
            dtype=dtype,
            device=device,
        )

        self.signal_std = signal_std
        self.noise_std = noise_std

        self.number_of_normal_samples = n_normal_samples
        normal = torch.distributions.Normal(loc=0, scale=1)
        self.uniform_normal_samples = normal.icdf(
            torch.Tensor(
                [
                    (1 / self.number_of_normal_samples) * x
                    for x in range(1, self.number_of_normal_samples)
                ]
            )
        )
        self.state = State(
            num_patients=self.evaluator.dim,
            num_tests_per_cycle=1,
            prior_infection_rate=prior_activeness_rate,
            noise_variance=self.noise_std**2,
            signal_variance=self.signal_std**2,
            device=device,
            dtype=dtype,
        )

        self.x = torch.empty(size=(0, self.evaluator.dim), dtype=dtype)
        self.fx = torch.empty(0, dtype=dtype)
        self.fx_noiseless = torch.empty(0, dtype=dtype)

        if evaluator.samples is not None:
            self.x = torch.vstack((self.x, self.evaluator.samples.cpu()))
            self.fx = torch.cat((self.fx, self.evaluator.values.cpu()))
            self.fx_noiseless = torch.cat(
                (self.fx_noiseless, self.evaluator.values_noiseless.cpu())
            )

        self.marginals = torch.empty(size=(0, self.evaluator.dim), dtype=dtype)
        self.groups = torch.empty(size=(0, self.evaluator.dim), dtype=torch.bool)
        self.converged = torch.zeros(self.evaluator.dim, dtype=torch.bool)
        self.mutual_informations = list()
        self.n_initial_groups = n_initial_groups
        self.lower_convergence_threshold = lower_convergence_threshold
        self.upper_convergence_threshold = upper_convergence_threshold
        self.prior_activeness_rate = prior_activeness_rate
        self.test_only_non_converged_lb = test_only_non_converged_lb
        self.test_only_non_converged_ub = test_only_non_converged_ub
        self.activeness_threshold = activeness_threshold
        self.results_dir = results_dir
        self.max_groups_per_iteration = max_groups_per_iteration
        self.maximum_mutual_information_loss = maximum_mutual_information_loss

        self.optimize_time = 0
        self.gamma_time = 0
        self.entropy_time = 0
        self.conditional_entropy_time = 0
        self.eval_time = 0
        self.produce_sample_time = 0
        self.resample_time = 0
        self.sample_times = []

        self.device = device
        self.dtype = dtype
        self.allowed = torch.ones(
            self.evaluator.dim, device=self.device, dtype=torch.int64
        )

    def optimize(self) -> State:
        """
        The main loop of the optimizer. Iteratively chooses a group G, evaluates it and updates the samples.

        Returns:
            The final state of the optimizer.
        """
        t_start = time.time()
        self.resample()
        new_marginals = torch.vstack(self.state.marginals)
        self.marginals = torch.vstack((self.marginals, new_marginals))
        test_results = []
        iteration = 0
        while iteration < self.gt_iterations:
            logging.debug(f"📌 Sampler iteration: {iteration}")

            new_groups = self.select_next_groups(
                self.max_groups_per_iteration, self.maximum_mutual_information_loss
            )
            for new_group in new_groups:
                new_group = new_group.unsqueeze(0)
                logging.debug(
                    f"👭 Next group to test:  {[i for i in range(new_group.shape[1]) if new_group.squeeze()[i]]}"
                )
                t_eval_start = time.time()
                tmp_default = None
                test_result, fval, fval_noiseless, config = self.evaluator(
                    torch.nonzero(new_group.squeeze()), tmp_default
                )
                test_result, fval, fval_noiseless, config = (
                    test_result.cpu(),
                    fval.cpu(),
                    fval_noiseless.cpu(),
                    config.cpu(),
                )
                test_results.append(test_result.item())
                self.eval_time += time.time() - t_eval_start
                self.x = torch.vstack((self.x, config.reshape(1, -1)))
                self.fx = torch.cat((self.fx, fval.reshape(-1)))
                self.fx_noiseless = torch.cat(
                    (self.fx_noiseless, fval_noiseless.reshape(-1))
                )
                self.state.add_test_results(test_result.reshape(-1))
                self.state.add_past_groups(new_group)
                self.groups = torch.vstack((self.groups, new_group))
                iteration += 1

            for ng in range(new_groups.shape[0] - 1):
                self.marginals = torch.vstack((self.marginals, new_marginals))

            self.resample()

            new_marginals = torch.vstack(self.state.marginals)
            self.marginals = torch.vstack((self.marginals, new_marginals))

            # write marginals to csv
            marginals_np = self.marginals.detach().cpu().numpy().squeeze()
            np.savetxt(
                os.path.join(self.results_dir, "marginals.csv"),
                marginals_np,
                delimiter=",",
            )
            np.savetxt(
                os.path.join(self.results_dir, "test_results.csv"),
                test_results,
                delimiter=",",
            )
            with lzma.open(
                os.path.join(self.results_dir, "tested_groups.csv.xz"), "w"
            ) as f:
                np.savetxt(f, self.state.past_groups.detach().cpu().numpy(), fmt="%d")

            self.converged = torch.logical_or(
                new_marginals.reshape(-1) <= self.lower_convergence_threshold,
                new_marginals.reshape(-1) >= self.upper_convergence_threshold,
            )

            marginals_string = ""
            for i, marginal in enumerate(
                new_marginals.detach().cpu().numpy().squeeze()
            ):
                marginals_string += f"{BColors.OKGREEN if self.converged[i] else ''}{marginal:.3f}{BColors.ENDC} "

            logging.info(
                f"🔎 Current marginals ({iteration} groups): {marginals_string}"
            )
            if (
                torch.all(self.converged)
                or iteration == self.gt_iterations
                or np.floor(iteration / 10) * 10 > iteration - new_groups.shape[0]
            ):
                # write marginals to csv
                marginals_np = self.marginals.detach().cpu().numpy().squeeze()
                np.savetxt(
                    os.path.join(self.results_dir, "marginals.csv"),
                    marginals_np,
                    delimiter=",",
                )
                np.savetxt(
                    os.path.join(self.results_dir, "test_results.csv"),
                    test_results,
                    delimiter=",",
                )
                with lzma.open(
                    os.path.join(self.results_dir, "results.csv.xz"), "w"
                ) as f:
                    np.savetxt(
                        f,
                        np.hstack(
                            [
                                self.x.detach().cpu().numpy(),
                                self.fx.detach().cpu().numpy().reshape(-1, 1),
                            ]
                        ),
                        delimiter=",",
                    )
                with lzma.open(
                    os.path.join(self.results_dir, "results_noiseless.csv.xz"), "w"
                ) as f:
                    np.savetxt(
                        f,
                        np.hstack(
                            [
                                self.x.detach().cpu().numpy(),
                                self.fx_noiseless.detach().cpu().numpy().reshape(-1, 1),
                            ]
                        ),
                        delimiter=",",
                    )

            # check convergence
            if torch.all(self.converged):
                logging.info("✅ Converged")
                break
            else:
                prev_marginals = new_marginals
        self.optimize_time = time.time() - t_start
        # self.sampler.update(new_group, test_result)
        return self.state

    @property
    def pure_noise_std(self) -> float:
        return self.evaluator.pure_noise_std

    def resample(self) -> None:
        """
        Resamples the particles using the SMC sampler.

        Returns:
            None.

        """
        self.state.marginals = []
        # compute marginal using SMC sampler

        sampler = self.sampler
        t_p_1 = time.time()
        sampler.produce_sample(self.state)
        t_p_2 = time.time()
        self.state.marginals.append(sampler.marginal)
        self.state.update_particles(sampler)
        t_p_3 = time.time()
        self.produce_sample_time += t_p_2 - t_p_1
        self.resample_time = t_p_3 - t_p_2
        self.sample_times.append(t_p_2 - t_p_1)

    def select_next_groups(
        self,
        max_groups: int = 1,
        max_loss: float = 0.1,
    ) -> torch.Tensor:
        """
        Selects the next group to test.

        Args:
            max_groups: the maximum number of groups to test
            max_loss: the maximum loss in mutual information to allow when selecting groups

        Returns:
            The next group(s) to test.

        """
        self.allowed = torch.logical_and(
            self.marginals[-1].reshape(-1) > self.test_only_non_converged_lb,
            self.marginals[-1].reshape(-1) < self.test_only_non_converged_ub,
        )

        best_groups = torch.Tensor()
        MIs = torch.Tensor()
        best_MI = 0
        while best_groups.shape[0] < max_groups:
            initial_groups = self.sample_initial_groups(
                self.evaluator.dim, self.n_initial_groups
            ).to(
                dtype=torch.int64,
            )
            best_group, MI = self.forward_backward(initial_groups.clone())
            best_MI = max(best_MI, MI)
            logging.debug(f"Best mutual information: {float(MI):.3f}")
            if (best_MI - MI) / best_MI > max_loss:
                break
            best_group = best_group.reshape(1, -1)
            best_groups = torch.cat((best_groups, best_group))
            MIs = torch.cat((MIs, torch.Tensor([MI])))
            self.allowed = torch.logical_and(
                self.allowed, torch.logical_not(best_group.squeeze())
            )
            best_groups = best_groups[(best_MI - MIs) / best_MI < max_loss]
            MIs = MIs[(best_MI - MIs) / best_MI < max_loss]

        return best_groups

    def sample_initial_groups(
        self,
        group_size: int,
        number_of_groups: int,
    ) -> torch.Tensor:
        """
        Samples initial groups to test.

        Args:
            group_size: the size of the groups
            number_of_groups: the number of groups to sample

        Returns:
            The sampled groups.

        """
        groups = torch.Tensor()
        for _ in range(number_of_groups):
            group = torch.zeros(group_size, dtype=torch.float64)
            indices = np.random.choice(
                [i for i in range(group_size) if self.allowed[i]],
                min(3, torch.sum(self.allowed).item()),
                replace=False,
            )
            group[indices] = 1
            groups = torch.cat((groups, group.unsqueeze(0)))
        return groups

    def forward_backward(
        self,
        initial_groups,
    ) -> Tuple[torch.Tensor, float]:
        """
        Performs a forward-backward pass to find the best group to test.

        Args:
            initial_groups: the initial groups to test

        Returns:
            The best group and its mutual information.

        """
        best_group = None
        best_mutual_information = -float("inf")
        self.mutual_informations.append([])  # for logging

        for G in initial_groups:
            best_round_mutual_information = self.calculate_mutual_information(G)
            mode = "fast_forwards" if self.evaluator.dim > 30 else "slow_forwards"
            dimensions_to_try = [d for d in range(self.evaluator.dim)]
            while True:
                if mode in ["fast_forwards", "slow_forwards"]:
                    best_idx, best_val, gains = self.forward_pass(G, dimensions_to_try)
                    if best_val > best_round_mutual_information:
                        G[best_idx] = 1
                        best_round_mutual_information = best_val
                        if mode == "fast_forwards":
                            dimensions_to_try = torch.topk(gains * (1 - G), 30).indices
                        else:
                            dimensions_to_try = [d for d in range(self.evaluator.dim)]
                    else:
                        if mode == "fast_forwards":
                            mode = "slow_forwards"
                            dimensions_to_try = [d for d in range(self.evaluator.dim)]
                        elif mode == "slow_forwards":
                            mode = "backwards"
                else:
                    best_idx, best_val = self.backward_pass(G)
                    if best_val > best_round_mutual_information:
                        G[best_idx] = 0
                        best_round_mutual_information = best_val
                        mode = "backwards_success"
                    else:
                        if mode == "backwards_success":
                            mode = "slow_forwards"
                        else:
                            break

            self.mutual_informations[-1].append(best_round_mutual_information)
            if best_round_mutual_information > best_mutual_information:
                best_mutual_information = best_round_mutual_information
                best_group = G

        return best_group, best_mutual_information

    def forward_pass(
        self,
        group,
        dimensions_to_try,
    ) -> Tuple[torch.Tensor, float, torch.Tensor]:
        """
        Performs a forward pass to find the best dimension to add to the group.

        Args:
            group: the group to test
            dimensions_to_try: the dimensions to try

        Returns:
            The best dimension to add to the group, the mutual information and the gains.

        """
        gain = torch.zeros(self.evaluator.dim)
        round_best_MI = -float("inf")
        best_idx = -1
        for d in dimensions_to_try:
            if self.allowed[d] and group[d] == 0:
                group[d] = 1
                mutual_information = self.calculate_mutual_information(group)
                gain[d] = mutual_information
                if mutual_information > round_best_MI:
                    round_best_MI = mutual_information
                    best_idx = d
                group[d] = 0

        return best_idx, round_best_MI, gain

    def backward_pass(
        self,
        group,
    ) -> Tuple[torch.Tensor, float]:
        """
        Performs a backward pass to find the best dimension to remove from the group.

        Args:
            group: the group to test

        Returns:
            The best dimension to remove from the group and the mutual information.

        """
        round_best_MI = -float("inf")
        best_idx = -1
        for d in range(self.evaluator.dim):
            if group[d] == 1:
                group[d] = 0
                mutual_information = self.calculate_mutual_information(group)
                if mutual_information > round_best_MI:
                    round_best_MI = mutual_information
                    best_idx = d
                group[d] = 1

        return best_idx, round_best_MI

    def calculate_mutual_information(self, group_to_test: torch.Tensor) -> float:
        """
        Calculates the mutual information for a given group.

        Args:
            group_to_test: the group to test

        Returns:
            The mutual information.

        """

        t1 = time.time()
        gamma = self.estimate_gamma(group_to_test)
        t2 = time.time()
        entropy = self.estimate_entropy(gamma)
        t3 = time.time()
        conditional_entropy = self.estimate_conditional_entropy(gamma)
        t4 = time.time()
        self.gamma_time += t2 - t1
        self.entropy_time += t3 - t2
        self.conditional_entropy_time += t4 - t3
        mutual_information = entropy - conditional_entropy

        return mutual_information

    def estimate_gamma(
        self, group_to_test: torch.Tensor, reduce_unique_particles: bool = False
    ) -> torch.Tensor:
        """
        Estimate the probability of a group of being active (gamma).

        Args:
            group_to_test: the group to test
            reduce_unique_particles: whether to reduce the unique particles

        Returns:
            The estimated probability of the group being active.

        """

        particles = self.state._particles_long

        if not reduce_unique_particles:
            unique_particles = particles
            unique_particle_weights = self.state.particle_weights
        else:
            unique_particles, inverse_mapping = torch.unique(
                particles, dim=0, return_inverse=True
            )
            unique_particle_weights = torch.zeros(
                unique_particles.shape[0], dtype=torch.float64
            )
            unique_particle_weights = unique_particle_weights.scatter_reduce(
                dim=0,
                index=inverse_mapping,
                src=self.state.particle_weights,
                reduce="sum",
            )

        positive_in_groups = (group_to_test.reshape(1, -1) @ unique_particles.T) > 0
        return torch.sum(positive_in_groups * unique_particle_weights)

    def estimate_entropy(
        self,
        gamma: torch.Tensor,
    ) -> float:
        """
        Estimate the entropy of the distribution.

        Args:
            gamma: the probability of a group being active

        Returns:
            the entropy

        """
        active_samples = self.uniform_normal_samples * self.signal_std
        inactive_samples = self.uniform_normal_samples * self.noise_std

        sigma_norm = Normal(0, self.signal_std)
        sigma_n_norm = Normal(0, self.noise_std)

        # CHECK
        entropy_active = torch.mean(
            torch.log(
                gamma * torch.exp(sigma_norm.log_prob(active_samples))
                + (1 - gamma) * torch.exp(sigma_n_norm.log_prob(active_samples))
            )
        )
        entropy_inactive = torch.mean(
            torch.log(
                gamma * torch.exp(sigma_norm.log_prob(inactive_samples))
                + (1 - gamma) * torch.exp(sigma_n_norm.log_prob(inactive_samples))
            )
        )

        entropy = -(gamma * entropy_active + (1 - gamma) * entropy_inactive)

        return entropy

    def estimate_conditional_entropy(
        self,
        gamma: torch.Tensor,
    ) -> float:
        """
        Estimate the conditional entropy for a given group.

        Args:
            gamma: the probability of the group being active

        Returns:
            the conditional entropy

        """
        conditional_entropy = 0.5 * (
            gamma * math.log(2 * self.signal_std**2 * math.pi * math.e)
            + (1 - gamma) * math.log(2 * self.noise_std**2 * math.pi * math.e)
        )
        return conditional_entropy
