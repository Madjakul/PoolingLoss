# pooling_loss/experiments/mteb_pooling_loss.py

from typing import TYPE_CHECKING, List, Union

import torch
import torch.nn.functional as F

from pooling_loss.modules import PoolingLoss
from pooling_loss.utils.helpers import get_tokenizer

if TYPE_CHECKING:
    from pooling_loss.utils.configs import BaseConfig


class MTEBPoolingLoss:

    def __init__(
        self,
        cfg: "BaseConfig",
        checkpoint_path: str,
    ) -> None:
        self.cfg = cfg

        self.model = PoolingLoss(cfg)

        # Determine device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["state_dict"])

        # Set up for inference
        self.model.to(self.device)
        self.model.eval()

        # Initialize tokenizer
        self.tokenizer = get_tokenizer(cfg.data.tokenizer_name)

    @torch.inference_mode()
    def encode(
        self,
        sentences: Union[List[str], str],
        **kwargs,
    ):
        if isinstance(sentences, str):
            sentences = [sentences]

        all_embeddings = []
        for start_index in range(0, len(sentences), self.cfg.data.batch_size):
            sentences_batch = sentences[
                start_index : start_index + self.cfg.data.batch_size
            ]

            # Tokenize batch
            inputs = self.tokenizer(
                sentences_batch,
                padding=self.cfg.data.padding,
                truncation=True,
                return_tensors="pt",
                max_length=self.cfg.data.max_length,
            ).to(self.device)

            # Get token embeddings from the model
            last_hidden_state = self.model(**inputs)

            # Perform mean pooling to get sentence embeddings
            attention_mask = inputs["attention_mask"]
            embeddings = (last_hidden_state * attention_mask.unsqueeze(-1)).sum(dim=1)
            embeddings = embeddings / attention_mask.sum(dim=1, keepdim=True).clamp(
                min=1e-9
            )
            normalized_embeddings = F.normalize(embeddings, p=2, dim=-1)

            all_embeddings.append(normalized_embeddings.cpu())

        return torch.cat(all_embeddings, dim=0).numpy()
