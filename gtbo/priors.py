import torch
from gpytorch.priors import LogNormalPrior


class CustomLogNormalPrior(LogNormalPrior):
    def __init__(self, loc, scale, validate_args=None, transform=None):
        super().__init__(
            loc=loc, scale=scale, validate_args=validate_args, transform=transform
        )

    def sample(self, sample_shape=torch.Size()):
        with torch.no_grad():
            if len(self.shape()) > 1:
                x = self.base_dist.sample()
            else:
                x = self.base_dist.sample(sample_shape)
            for transform in self.transforms:
                x = transform(x)
            return x
