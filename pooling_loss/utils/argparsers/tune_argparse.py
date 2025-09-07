# pooling_loss/utils/argparsers/tune_argparse.py

import argparse
from typing import List


class TuneArgparse:

    @classmethod
    def parse_known_args(cls):
        parser = argparse.ArgumentParser(
            description="Arguments used for hyper-parameter tuning."
        )
        parser.add_argument(
            "--config_path",
            type=str,
            required=True,
            help="Path to the config file.",
        )
        parser.add_argument(
            "--ray_storage_path",
            type=str,
            required=True,
            help="Directory where Ray will save the logs and experiments results.",
        )
        parser.add_argument(
            "--logs_dir",
            type=str,
            required=True,
            help="Directory where the logs will be saved.",
        )
        parser.add_argument(
            "--processed_ds_path",
            type=str,
            required=True,
            help="Path to save/load the prprocessed dataset.",
        )
        parser.add_argument(
            "--individual_processed_paths",
            nargs="+",  # Expect 1 or more arguments and gather them into a list
            type=str,
            default=[],
            help=(
                "List of paths to the individual processed datasets (for unified"
                " tuning). Not required if the unified dataset has already been"
                " processed."
            ),
        )
        parser.add_argument(
            "--num_proc",
            type=int,
            default=None,
            help="Number of processes to use. Default is the number of CPUs minus one.",
        )
        parser.add_argument(
            "--cache_dir",
            type=str,
            default=None,
            help="Path to the cache directory for HuggingFace.",
        )
        args, _ = parser.parse_known_args()
        return args
