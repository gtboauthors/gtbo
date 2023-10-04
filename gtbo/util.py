import os
import subprocess
from typing import List

import numpy as np


def to_unit_cube(x, lb, ub):
    """
    Project to [0, 1]^d from hypercube with bounds lb and ub

    Args:
        x: the point to map to the unit cube
        lb: the lower bound in the original space
        ub: the upper bound in the original space

    Returns:
        the point in the unit cube

    """
    assert lb.ndim == 1 and ub.ndim == 1
    xx = (x - lb) / (ub - lb)
    return xx


def from_unit_cube(x, lb, ub):
    """Project from [0, 1]^d to hypercube with bounds lb and ub"""
    assert lb.ndim == 1 and ub.ndim == 1
    xx = x * (ub - lb) + lb
    return xx


def logit(x: float):
    """
    The logit function

    Args:
        x: parameter

    Returns:
        the logit

    """
    return np.log(x / (1 - x))


def eval_singularity_benchmark(
    eval_points: List[List[float]], singularity_image_path: str, name: str
) -> float:
    cmd = (
        f"$( cd {singularity_image_path} ; poetry env info --path)/bin/python3 {os.path.join(singularity_image_path, 'main.py')} --name {name} "
        f"-x {' '.join(list(map(lambda _x: str(_x), eval_points)))}"
    )
    process = subprocess.check_output(
        cmd,
        shell=True,
        env={
            **os.environ,
            **{
                "LD_LIBRARY_PATH": f"{singularity_image_path}/data/mujoco210/bin:/usr/lib/nvidia",
                "MUJOCO_PY_MUJOCO_PATH": f"{singularity_image_path}/data/mujoco210",
            },
        },
    )
    res = process.decode().split("\n")[-2]
    return float(res)
