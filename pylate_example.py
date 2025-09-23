# pylate_example.py

import os

import torch
from datasets import load_dataset
from pylate import evaluation, losses, models, utils
from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)

os.environ["WANDB_PROJECT"] = "pooling-loss"

# Define model parameters for contrastive training
model_name = "bert-base-uncased"  # Choose the pre-trained model you want to use as base
batch_size = 32  # Larger batch size often improves results, but requires more memory

num_train_epochs = 1  # Adjust based on your requirements
# Set the run name for logging and output directory
run_name = "contrastive-bert-base-uncased"
output_dir = f"output/{run_name}"

# 1. Here we define our ColBERT model. If not a ColBERT model, will add a linear layer to the base encoder.
model = models.ColBERT(model_name_or_path=model_name)

# Compiling the model makes the training faster
model = torch.compile(model)

# Load dataset
dataset = load_dataset("sentence-transformers/msmarco-bm25", "triplet", split="train")
# Split the dataset (this dataset does not have a validation set, so we split the training set)
splits = dataset.train_test_split(test_size=0.01)
train_dataset = splits["train"]
eval_dataset = splits["test"]

# Define the loss function
train_loss = losses.Contrastive(model=model)

# Initialize the evaluator
dev_evaluator = evaluation.ColBERTTripletEvaluator(
    anchors=eval_dataset["query"],
    positives=eval_dataset["positive"],
    negatives=eval_dataset["negative"],
)

# Configure the training arguments (e.g., batch size, evaluation strategy, logging steps)
args = SentenceTransformerTrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    eval_strategy="steps",
    eval_steps=250,
    logging_steps=1,
    fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=False,  # Set to True if you have a GPU that supports BF16
    run_name=run_name,  # Will be used in W&B if `wandb` is installed
    learning_rate=2e-5,
    report_to="wandb",
)

# Initialize the trainer for the contrastive training
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=train_loss,
    evaluator=dev_evaluator,
    data_collator=utils.ColBERTCollator(model.tokenize),
)
# Start the training process
trainer.train()


# pylate_example.py

# import os
#
# import torch
# from datasets import load_dataset
# from pylate import evaluation, losses, models, scores, utils
# from sentence_transformers import (
#     SentenceTransformerTrainer,
#     SentenceTransformerTrainingArguments,
# )
# from torch import Tensor
#
#
# # This is a helper function that might be in a different module in the actual library
# def convert_to_tensor(X):
#     if isinstance(X, Tensor):
#         return X
#     return torch.tensor(X)
#
#
# # <<< 1. DEFINE YOUR CUSTOM SCORING FUNCTION HERE
# def colbert_scores_inf_mask(
#     queries_embeddings: Tensor,
#     documents_embeddings: Tensor,
#     queries_mask: Tensor = None,
#     documents_mask: Tensor = None,
# ) -> Tensor:
#     """Computes ColBERT scores using -inf masking for padded document
#     tokens."""
#     queries_embeddings = convert_to_tensor(queries_embeddings)
#     documents_embeddings = convert_to_tensor(documents_embeddings)
#     scores = torch.einsum(
#         "ash,bth->abst",
#         queries_embeddings,
#         documents_embeddings,
#     )
#     # Mask padded DOCUMENT tokens with -inf before the max operation
#     if documents_mask is not None:
#         documents_mask = convert_to_tensor(documents_mask)
#         expanded_doc_mask = documents_mask.unsqueeze(0).unsqueeze(2).bool()
#
#         # <<< MODIFIED LINE HERE
#         # Replaced torch.finfo with float("-inf") as requested.
#         scores = scores.masked_fill(~expanded_doc_mask, float("-inf"))
#
#     # Max-pool over the document tokens
#     scores = scores.max(axis=-1).values
#     # Nullify scores from padded QUERY tokens
#     if queries_mask is not None:
#         queries_mask = convert_to_tensor(queries_mask)
#         scores = scores * queries_mask.unsqueeze(1).float()
#     # Sum over the query tokens
#     scores = scores.sum(axis=-1)
#     return scores
#
#
# os.environ["WANDB_PROJECT"] = "pooling-loss"
#
# # Define model parameters
# model_name = "bert-base-uncased"
# batch_size = 32
# num_train_epochs = 1
# run_name = "contrastive-bert-base-uncased-inf-mask"
# output_dir = f"output/{run_name}"
#
# # Define ColBERT model
# scores.colbert_scores = colbert_scores_inf_mask
# model = models.ColBERT(model_name_or_path=model_name)
# model = torch.compile(model)
#
# # Load and split dataset
# dataset = load_dataset("sentence-transformers/msmarco-bm25", "triplet", split="train")
# splits = dataset.train_test_split(test_size=0.01)
# train_dataset = splits["train"]
# eval_dataset = splits["test"]
#
# # Define the loss function
# train_loss = losses.Contrastive(model=model)
#
# # REASSIGN THE SCORING FUNCTION HERE
# train_loss.score_metric = colbert_scores_inf_mask
# print("Successfully replaced the default score metric with 'colbert_scores_inf_mask'.")
#
# # Initialize the evaluator
# dev_evaluator = evaluation.ColBERTTripletEvaluator(
#     anchors=eval_dataset["query"],
#     positives=eval_dataset["positive"],
#     negatives=eval_dataset["negative"],
# )
#
# # Configure training arguments
# args = SentenceTransformerTrainingArguments(
#     output_dir=output_dir,
#     num_train_epochs=num_train_epochs,
#     per_device_train_batch_size=batch_size,
#     per_device_eval_batch_size=batch_size,
#     eval_strategy="steps",
#     eval_steps=250,
#     logging_steps=1,
#     fp16=True,
#     bf16=False,
#     run_name=run_name,
#     learning_rate=2e-5,
#     report_to="wandb",
# )
#
# # Initialize the trainer
# trainer = SentenceTransformerTrainer(
#     model=model,
#     args=args,
#     train_dataset=train_dataset,
#     eval_dataset=eval_dataset,
#     loss=train_loss,
#     evaluator=dev_evaluator,
#     data_collator=utils.ColBERTCollator(model.tokenize),
# )
#
# # Start training
# trainer.train()
