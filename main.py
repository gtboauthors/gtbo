import argparse
import logging

import gin

from gtbo.group_testing.utils import BColors
from gtbo.gtbo import GTBO

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="GTBO",
        description="Group-Testing Bayesian Optimization",
        epilog="For more information, please contact the author.",
    )

    parser.add_argument(
        "--gin-files",
        type=str,
        nargs="+",
        default=["configs/default.gin"],
        help="Path to the config file",
    )
    parser.add_argument(
        "--gin-bindings",
        type=str,
        nargs="+",
        default=[],
    )

    args = parser.parse_args()

    gin.parse_config_files_and_bindings(args.gin_files, args.gin_bindings)

    logging.basicConfig(
        level=gin.query_parameter("GTBO.logging_level"),
        format=f"{BColors.LIGHTGREY} %(levelname)s:%(asctime)s - (%(filename)s:%(lineno)d) - %(message)s {BColors.ENDC}",
    )

    runner = GTBO()
    runner.run()
    gin.clear_config()
