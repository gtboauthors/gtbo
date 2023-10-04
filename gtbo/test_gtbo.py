import gin
import torch

from gtbo.benchmarks import BraninEffectiveDim
from gtbo.gtbo import GTBO


def test_runnability():
    gin.bind_parameter('GroupTester.gt_iterations', 2)
    gin.bind_parameter('GroupTester.activeness_threshold', 0.01)
    gin.bind_parameter('GroundTruthEvaluator.n_default_samples', 10)
    gin.bind_parameter('get_gp.active_prior_parameters', (0, 1))
    gin.bind_parameter('get_gp.inactive_prior_parameters', (3, 1))

    gtbo = GTBO(
        benchmark=BraninEffectiveDim(dim=50, dtype=torch.double),
        maximum_number_evaluations=10,
        number_initial_points=3,
        results_dir="results",
    )
    gtbo.run()
