import torch

# -- 1. Setup: Create small tensors for our test --

# A mask to apply
valid_mask = torch.tensor([[True, True, False], [True, False, True]], dtype=torch.bool)

# A large negative value for max-pooling
IGNORE = -1e9

# -- 2. Define the two methods we want to compare --


def calculate_scores_inplace(sim_matrix, mask):
    """Calculates scores by modifying the sim_matrix in-place."""
    print("\n--- Running In-Place Method ---")

    # --- The key insight: in-place operations on intermediate tensors ---
    # The error occurs on leaf variables, but sim_matrix here is intermediate.

    # a. Calculate mean
    sim_matrix.masked_fill_(~mask, 0.0)
    expected_scores = sim_matrix.sum() / mask.sum()

    # b. Calculate max-sum
    sim_matrix.masked_fill_(~mask, IGNORE)
    summed_max_sim = sim_matrix.max()  # Simplified for this example

    return summed_max_sim - expected_scores


def calculate_scores_copy(sim_matrix, mask):
    """Calculates scores by creating copies to avoid in-place changes."""
    print("\n--- Running Copy Method ---")

    # a. Calculate mean on a copy
    sim_for_mean = sim_matrix.clone().masked_fill_(~mask, 0.0)
    expected_scores = sim_for_mean.sum() / mask.sum()

    # b. Calculate max-sum on another copy
    sim_for_max = sim_matrix.clone().masked_fill_(~mask, IGNORE)
    summed_max_sim = sim_for_max.max()

    return summed_max_sim - expected_scores


# -- 3. Execute and Verify --

# --- IN-PLACE TEST ---
# Create the leaf tensor first
leaf_tensor_inplace = torch.tensor(
    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
)
# Create the intermediate tensor. The in-place operations will happen on this.
# A simple identity operation makes it intermediate.
intermediate_matrix_inplace = leaf_tensor_inplace * 1.0

final_score_inplace = calculate_scores_inplace(intermediate_matrix_inplace, valid_mask)
print(f"Final Score (In-place): {final_score_inplace.item()}")
final_score_inplace.backward()
grad_inplace = leaf_tensor_inplace.grad.clone()  # Check gradient on the leaf

# --- COPY TEST ---
leaf_tensor_copy = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
intermediate_matrix_copy = leaf_tensor_copy * 1.0

final_score_copy = calculate_scores_copy(intermediate_matrix_copy, valid_mask)
print(f"Final Score (Copy): {final_score_copy.item()}")
final_score_copy.backward()
grad_copy = leaf_tensor_copy.grad

# -- 4. Compare the results --
print("\n--- Verification ---")
print(f"Gradient from In-place method:\n{grad_inplace}")
print(f"Gradient from Copy method:\n{grad_copy}")

are_grads_same = torch.allclose(grad_inplace, grad_copy)
print(f"\nVerification successful: {are_grads_same}")

if are_grads_same:
    print("✅ The script ran without errors and the gradients are identical.")
else:
    print("❌ The gradients are different or an error occurred.")
