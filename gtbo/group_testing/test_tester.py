import torch

from gtbo.benchmarks import HartmannEffectiveDim
from gtbo.group_testing.ground_truth_evaluator import GroundTruthEvaluator
from gtbo.group_testing.tester import GroupTester


def test_estimate_gamma():
    group_tester = GroupTester(
        evaluator=GroundTruthEvaluator(
            benchmark=HartmannEffectiveDim(
                dim=10
            ),
            default=torch.zeros(10, dtype=torch.double),
            n_default_samples=2,
        ),
        noise_std=0.1,
        signal_std=0.1,
    )
    group_tester.state._particles_long = torch.tensor(
        [[1, 0], [0, 1], [0, 1]], dtype=torch.int64
    )

    group_tester.state.particle_weights = torch.tensor(
        [0.3, 0.6, 0.1], dtype=torch.double
    )

    gamma1 = group_tester.estimate_gamma(
        group_to_test=torch.tensor([1, 0], dtype=torch.int64)
    )

    assert gamma1 == 0.3

    gamma2 = group_tester.estimate_gamma(
        group_to_test=torch.tensor([0, 1], dtype=torch.int64)
    )
    assert gamma2 == 0.7
