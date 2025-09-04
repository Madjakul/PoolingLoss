# pooling_loss/callbacks/pe_pca.py

import logging
import os.path as osp

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from lightning.pytorch.callbacks import Callback


class PositionalEmbeddingPCA(Callback):

    def __init__(self, output_dir: str = "pca_analysis"):
        super().__init__()
        self.output_dir = output_dir

    def _perform_torch_pca(self, tensor: torch.Tensor, k: int = 2) -> torch.Tensor:
        """Performs PCA on the GPU using SVD."""
        # 1. Center the data
        mean = torch.mean(tensor, dim=0, keepdim=True)
        centered_tensor = tensor - mean

        # 2. Perform SVD
        _, _, Vh = torch.linalg.svd(centered_tensor)

        # 3. Project the data onto the top k principal components
        # V is the matrix of principal components (right singular vectors)
        V = Vh.mH  # Transpose for real matrices
        projected_tensor = torch.matmul(centered_tensor, V[:, :k])

        return projected_tensor

    def on_validation_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        """Called when the validation epoch ends."""
        if trainer.state.fn != "fit":
            return

        try:
            # Keep the tensor on the same device as the model (GPU)
            positional_embeddings = (
                pl_module.lm.model.roberta.embeddings.position_embeddings.weight.clone().detach()
            )

            # --- Perform PCA on GPU ---
            pca_result_gpu = self._perform_torch_pca(positional_embeddings, k=2)

            # Move to CPU only for saving and plotting
            pca_result_cpu = pca_result_gpu.cpu().numpy()

            # --- Save the raw data points ---
            epoch = trainer.current_epoch
            step = trainer.global_step
            data_path = osp.join(
                self.output_dir, f"pca_coords_epoch={epoch}_step={step}.csv"
            )
            np.savetxt(
                data_path, pca_result_cpu, delimiter=",", header="PC1,PC2", comments=""
            )

            # --- Save a vector figure for quick inspection ---
            plt.figure(figsize=(10, 8))
            sns.scatterplot(
                x=pca_result_cpu[:, 0],
                y=pca_result_cpu[:, 1],
                hue=range(len(pca_result_cpu)),
                palette="viridis",
                legend="auto",
            )
            plt.title(f"PCA of Positional Embeddings - Epoch {epoch}, Step {step}")
            plt.xlabel("Principal Component 1")
            plt.ylabel("Principal Component 2")

            # Save as PDF for high-quality, scalable figures
            plot_path = osp.join(
                self.output_dir, f"pca_plot_epoch={epoch}_step={step}.pdf"
            )
            plt.savefig(plot_path, format="pdf", bbox_inches="tight")
            plt.close()

        except AttributeError:
            logging.error("Could not find positional embeddings for PCA plot.")
        except Exception as e:
            logging.error(f"Failed to create/save PCA analysis: {e}")
