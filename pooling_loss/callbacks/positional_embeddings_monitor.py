import torch
from lightning.pytorch.callbacks import Callback


class PositionalEmbeddingsMonitor(Callback):
    """Measures how much positional signal is present in the final
    embeddings."""

    def __init__(self):
        super().__init__()
        self.similarities = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Assumes your model is named 'model' within the LightningModule
        # and has a standard encoder-decoder structure. Adjust as needed.
        encoder = pl_module.model.get_encoder()

        # 1. Get final hidden states
        # You might need to run a forward pass here or get it from `outputs`
        # For simplicity, let's assume `outputs` contains the last_hidden_state
        last_hidden_state = (
            outputs.last_hidden_state
        )  # Shape: (batch, seq_len, hidden_dim)

        # 2. Get raw positional embeddings
        # This requires accessing the embedding layer directly
        position_ids = torch.arange(
            last_hidden_state.size(1), device=pl_module.device
        ).expand((last_hidden_state.size(0), -1))
        positional_embeddings = encoder.embed_positions(
            position_ids
        )  # Shape: (batch, seq_len, hidden_dim)

        # 3. Calculate cosine similarity
        # We normalize the vectors (L2 norm) before the dot product
        similarity = torch.nn.functional.cosine_similarity(
            last_hidden_state, positional_embeddings, dim=-1
        )

        # 4. Store the mean similarity for this batch
        self.similarities.append(similarity.mean().item())

    def on_validation_epoch_end(self, trainer, pl_module):
        if not self.similarities:
            return

        avg_similarity = sum(self.similarities) / len(self.similarities)
        pl_module.log("avg_positional_signal", avg_similarity, on_epoch=True)
        self.similarities.clear()
