# pooling_loss/utils/argparsers/test_argparse.py

import argparse


class TestArgparse:

    @classmethod
    def parse_known_args(cls):
        parser = argparse.ArgumentParser(description="Arguments used to test a model.")
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
            "--checkpoint_path",
            type=str,
            required=True,
            help="Path to the model's checkpoint.",
        )
        parser.add_argument(
            "--processed_ds_path",
            type=str,
            required=True,
            help="Path to save/load the prprocessed dataset.",
        )
        parser.add_argument(
            "--num_proc",
            type=int,
            default=1,
            help="Number of processes to use. Default is the number of CPUs.",
        )
        parser.add_argument(
            "--cache_dir",
            type=str,
            default=None,
            help="Path to the cache directory for HuggingFace.",
        )
        args, _ = parser.parse_known_args()
        return args
