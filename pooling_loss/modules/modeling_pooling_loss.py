# pooling_loss/modules/modeling_pooling_loss.py

import logging
from typing import TYPE_CHECKING, Any, Dict

import lightning as L
import torch
from jaxtyping import Float, Int
from torcheval.metrics import BinaryAUROC, HitRate, ReciprocalRank
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    get_constant_schedule_with_warmup,
)

from pooling_loss.modules.alignment_uniformity_loss import AlignmentUniformityLoss
from pooling_loss.modules.info_nce_loss import InfoNCELoss
from pooling_loss.modules.pairwise_ce_loss import PairwiseCELoss
from pooling_loss.modules.triplet_loss import TripletLoss

if TYPE_CHECKING:
    from pooling_loss.utils.configs import BaseConfig


class PoolingLoss(L.LightningModule):

    loss_map = {
        "info-nce": InfoNCELoss,
        "triplet": TripletLoss,
        "pairwise-ce": PairwiseCELoss,
    }
    val_auroc: BinaryAUROC
    val_hr1: HitRate
    val_hr5: HitRate
    val_hr10: HitRate
    val_rr: ReciprocalRank
    test_auroc: BinaryAUROC
    test_hr1: HitRate
    test_hr5: HitRate
    test_hr10: HitRate
    test_rr: ReciprocalRank

    def __init__(self, cfg: "BaseConfig") -> None:
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg

        config = AutoConfig.from_pretrained(self.cfg.model.base_model_name)

        if self.cfg.model.is_decoder_model:
            logging.info(
                f"Loading pretrained decoder from {self.cfg.model.base_model_name}."
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.cfg.model.base_model_name, config=config
            )
        else:
            logging.info(
                f"Loading pretrained encoder from {self.cfg.model.base_model_name}."
            )
            self.model = AutoModelForMaskedLM.from_pretrained(
                self.cfg.model.base_model_name, config=config
            )
        self.hidden_size = self.model.config.hidden_size
        self.vocab_size = self.model.config.vocab_size

        self.contrastive_loss = self.loss_map[cfg.execution.loss](cfg)
        self.alignment_uniformity_loss = AlignmentUniformityLoss()

    def configure_optimizers(self) -> Dict[str, Any]:  # type: ignore[override]
        logging.info(
            f"""Configuring optimizer: AdamW with lr={self.cfg.execution.lr},
             weight_decay={self.cfg.execution.weight_decay}."""
        )

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.execution.lr,  # type: ignore
            weight_decay=self.cfg.execution.weight_decay,  # type: ignore
        )
        # Calculate steps dynamically
        total_steps = int(self.trainer.estimated_stepping_batches)
        warmup_steps = max(1, int(0.1 * total_steps))

        scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            last_epoch=-1,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def forward(
        self,
        input_ids: Int[torch.Tensor, "batch seq"],
        attention_mask: Int[torch.Tensor, "batch seq"],
    ) -> Float[torch.Tensor, "batch seq hidden"]:
        out = self.model(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden_states = out.hidden_states[-1]
        return last_hidden_states

    def training_step(self, batch, batch_idx: int) -> Float[torch.Tensor, ""]:
        q_embs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        pos_embs = self(
            input_ids=batch["pos_input_ids"],
            attention_mask=batch["pos_attention_mask"],
        )
        neg_embs = self(
            input_ids=batch["neg_input_ids"],
            attention_mask=batch["neg_attention_mask"],
        )

        k_embs = torch.cat([pos_embs, neg_embs], dim=0)  # (2B, S, H)
        k_mask = torch.cat(
            [batch["pos_attention_mask"], batch["neg_attention_mask"]],
            dim=0,
        )  # (2B, S)

        loss_metrics = self.contrastive_loss(
            query_embs=q_embs,
            key_embs=k_embs,
            q_mask=batch["attention_mask"],
            k_mask=k_mask,
        )
        alignment_uniformity_metrics = self.alignment_uniformity_loss(
            query_embs=q_embs,
            key_embs=k_embs,
            q_mask=batch["attention_mask"],
            k_mask=k_mask,
        )

        self.log_dict(
            {
                "train_alignment_loss": alignment_uniformity_metrics["alignment_loss"],
                "train_uniformity_loss": alignment_uniformity_metrics[
                    "uniformity_loss"
                ],
            },
            prog_bar=True,
            on_step=True,
            on_epoch=False,
            sync_dist=False,
            batch_size=self.cfg.data.batch_size,
        )
        self.log(
            "loss",
            loss_metrics["loss"],
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=self.cfg.data.batch_size,
        )
        return loss_metrics["loss"]

    def on_validation_start(self):
        """Move validation metrics to correct device before validation."""
        self.val_auroc = BinaryAUROC(device=self.device)
        self.val_hr1 = HitRate(k=1, device=self.device)
        self.val_hr5 = HitRate(k=5, device=self.device)
        self.val_hr10 = HitRate(k=10, device=self.device)
        self.val_rr = ReciprocalRank(device=self.device)

    def validation_step(self, batch, batch_idx: int) -> None:
        q_embs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        pos_embs = self(
            input_ids=batch["pos_input_ids"],
            attention_mask=batch["pos_attention_mask"],
        )
        neg_embs = self(
            input_ids=batch["neg_input_ids"],
            attention_mask=batch["neg_attention_mask"],
        )

        k_embs = torch.cat([pos_embs, neg_embs], dim=0)  # (2B, S, H)
        k_mask = torch.cat(
            [batch["pos_attention_mask"], batch["neg_attention_mask"]],
            dim=0,
        )  # (2B, S)

        loss_metrics = self.contrastive_loss(
            query_embs=q_embs,
            key_embs=k_embs,
            q_mask=batch["attention_mask"],
            k_mask=k_mask,
        )
        alignment_uniformity_metrics = self.alignment_uniformity_loss(
            query_embs=q_embs,
            key_embs=k_embs,
            q_mask=batch["attention_mask"],
            k_mask=k_mask,
        )

        all_scores = loss_metrics["all_scores"]
        targets = loss_metrics["targets"]
        poss = loss_metrics["poss"]
        negs = loss_metrics["negs"]
        batch_size = targets.size(0)
        binary_scores = torch.cat([poss, negs], dim=0)
        labels = torch.cat(
            [torch.ones(batch_size), torch.zeros(batch_size)], dim=0
        ).long()

        self.val_auroc.update(binary_scores, labels)
        self.val_hr1.update(all_scores, targets)
        self.val_hr5.update(all_scores, targets)
        self.val_hr10.update(all_scores, targets)
        self.val_rr.update(all_scores, targets)

        self.log_dict(
            {
                "val_alignment_loss": alignment_uniformity_metrics["alignment_loss"],
                "val_uniformity_loss": alignment_uniformity_metrics["uniformity_loss"],
            },
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=self.cfg.data.batch_size,
        )

    def on_validation_epoch_end(self) -> None:
        self.log("completed_epoch", self.current_epoch, prog_bar=False)
        auroc = self.val_auroc.compute()
        avg_hr1 = self.val_hr1.compute().mean()
        avg_hr5 = self.val_hr5.compute().mean()
        avg_hr10 = self.val_hr10.compute().mean()
        mrr = self.val_rr.compute().mean()
        self.log_dict(
            {
                "val_auroc": auroc,
                "val_hr1": avg_hr1,
                "val_hr5": avg_hr5,
                "val_hr10": avg_hr10,
                "val_mrr": mrr,
            },
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.val_auroc.reset()
        self.val_hr1.reset()
        self.val_hr5.reset()
        self.val_hr10.reset()
        self.val_rr.reset()

    def on_test_start(self) -> None:
        # this.current_device is now available
        self.test_auroc = BinaryAUROC(device=self.device)
        self.test_hr1 = HitRate(k=1, device=self.device)
        self.test_hr5 = HitRate(k=5, device=self.device)
        self.test_hr10 = HitRate(k=10, device=self.device)
        self.test_rr = ReciprocalRank(device=self.device)

    def test_step(self, batch, batch_idx: int) -> None:
        q_embs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        pos_embs = self(
            input_ids=batch["pos_input_ids"],
            attention_mask=batch["pos_attention_mask"],
        )
        neg_embs = self(
            input_ids=batch["neg_input_ids"],
            attention_mask=batch["neg_attention_mask"],
        )

        k_embs = torch.cat([pos_embs, neg_embs], dim=0)  # (2B, S, H)
        k_mask = torch.cat(
            [batch["pos_attention_mask"], batch["neg_attention_mask"]],
            dim=0,
        )  # (2B, S)

        loss_metrics = self.contrastive_loss(
            query_embs=q_embs,
            key_embs=k_embs,
            q_mask=batch["attention_mask"],
            k_mask=k_mask,
        )
        alignment_uniformity_metrics = self.alignment_uniformity_loss(
            query_embs=q_embs,
            key_embs=k_embs,
            q_mask=batch["attention_mask"],
            k_mask=k_mask,
        )

        all_scores = loss_metrics["all_scores"]
        targets = loss_metrics["targets"]
        poss = loss_metrics["poss"]
        negs = loss_metrics["negs"]
        batch_size = targets.size(0)
        binary_scores = torch.cat([poss, negs], dim=0)
        labels = torch.cat(
            [torch.ones(batch_size), torch.zeros(batch_size)], dim=0
        ).long()

        self.test_auroc.update(binary_scores, labels)
        self.test_hr1.update(all_scores, targets)
        self.test_hr5.update(all_scores, targets)
        self.test_hr10.update(all_scores, targets)
        self.test_rr.update(all_scores, targets)

        self.log_dict(
            {
                "test_alignment_loss": alignment_uniformity_metrics["alignment_loss"],
                "test_uniformity_loss": alignment_uniformity_metrics["uniformity_loss"],
            },
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=self.cfg.data.batch_size,
        )

    def on_test_epoch_end(self) -> None:
        auroc = self.test_auroc.compute()
        avg_hr1 = self.test_hr1.compute().mean()
        avg_hr5 = self.test_hr5.compute().mean()
        avg_hr10 = self.test_hr10.compute().mean()
        mrr = self.test_rr.compute().mean()
        self.log_dict(
            {
                "test_auroc": auroc,
                "test_hr1": avg_hr1,
                "test_hr5": avg_hr5,
                "test_hr10": avg_hr10,
                "test_mrr": mrr,
            },
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
