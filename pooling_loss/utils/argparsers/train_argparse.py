# pooling_loss/utils/argparsers/train_argparse.py

import argparse
from typing import List


class TrainArgparse:

    @classmethod
    def parse_known_args(cls):
        """Parses arguments.

        Returns
        -------
        args: Any
            Parsed arguments.
        """
        parser = argparse.ArgumentParser(
            description="Arguments used to train/fine-tune a model."
        )
        parser.add_argument(
            "--config_path",
            type=str,
            required=True,
            help="Path to the config file.",
        )
        parser.add_argument(
            "--logs_dir",
            type=str,
            required=True,
            help="Directory where the logs are stored.",
        )
        parser.add_argument(
            "--processed_ds_path",
            type=str,
            required=True,
            help="Path to save/load the prprocessed dataset.",
        )
        parser.add_argument(
            "--individual_processed_paths",
            type=List[str],
            default=[],
            help=(
                "List of paths to the individual processed datasets (for unified"
                " training). Not required if the unified dataset has already been"
                " processed."
            ),
        )
        parser.add_argument(
            "--num_proc",
            type=int,
            default=1,
            help="Number of processes to use. Default is the number of CPUs.",
        )
        parser.add_argument(
            "--checkpoint_dir",
            type=str,
            default=None,
            help="Directory where the model's checkpoints are stored.",
        )
        parser.add_argument(
            "--cache_dir",
            type=str,
            default=None,
            help="Path to the cache directory for HuggingFace.",
        )
        args, _ = parser.parse_known_args()
        return args
