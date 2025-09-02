# pooling_loss/utils/helpers.py

from typing import Any

from transformers import AutoTokenizer, PreTrainedTokenizerBase

WIDTH = 88


def flatten_mteb_results(results):
    flat_results = {}
    for task_name, task_results in results.items():
        for split_name, split_results in task_results.items():
            for metric_name, metric_value in split_results.items():
                key = f"{task_name}/{split_name}/{metric_name}"
                flat_results[key] = metric_value
    return flat_results


def get_tokenizer(model_name: str, **kwargs) -> "PreTrainedTokenizerBase":
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            raise ValueError("Tokenizer has neither pad_token nor eos_token defined.")
    return tokenizer


class DictAccessMixin:

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)
