"""
Demo for LieBN on Correlation Matrices

@author: Ziheng Chen
Please cite:

Ziheng Chen, Yue Song, Yunmei Liu, Nicu Sebe.
A Lie Group Approach to Riemannian Batch Normalization. ICLR 2024

Copyright (C) 2024 Ziheng Chen
All rights reserved.
"""

import torch as th
import geoopt
from LieBN import LieBNCor
from LieBN.Geometry.Correlation import Correlation

# Set the random seed for reproducibility
SEED = 42
th.manual_seed(SEED)


# Helper function to generate random correlation matrices
def random_correlation_matrix(bs, c, n):
    corr_module = Correlation(n)
    return corr_module.random_cor([bs, c, n, n]).to(th.double)


# Set dimensions
bs, c, n = 4, 2, 5  # Batch size, channels, matrix size
shape = [c, n, n]
manifold = Correlation(n=n)

# Instantiate correlation metric module and generate random input
P = manifold.random(bs, c, n,n).requires_grad_(True).to(th.double)  # Input correlation matrices
target = manifold.random(bs, c, n,n).to(th.double)  # Target correlation matrices for loss computation

# Instantiate LieBN for Correlation Manifold (ECM, LECM, OLM, LSM)
liebn = LieBNCor(shape=shape, metric='OLM',batchdim=[0]).to(th.double)

print("\nLieBNCor Layer Initialized:", liebn)
print("Manifold:", liebn.manifold)

print("\n=== LieBNCor Parameters ===")
for name, param in liebn.named_parameters():
    print(f"Parameter Name: {name}, Shape: {param.shape}, Requires Grad: {param.requires_grad}")

# Define optimizer and loss function
optimizer = geoopt.optim.RiemannianAdam(liebn.parameters(), lr=1e-3)
criterion = th.nn.MSELoss()

# Print initial loss
with th.no_grad():
    initial_loss = criterion(liebn(P), target).item()
print(f"\nInitial Loss: {initial_loss:.6f}")

# Training loop
num_epochs = 5  # Set higher if needed
for i in range(num_epochs):
    liebn.train()  # Set to training mode
    optimizer.zero_grad()  # Clear previous gradients
    output = liebn(P)
    # Compute loss
    loss = criterion(output, target)

    # Backpropagation
    loss.backward()

    # Gradient Norm Check (Optional)
    grad_norm = th.nn.utils.clip_grad_norm_(liebn.parameters(), max_norm=1.0)

    optimizer.step()

    # Print loss every iteration
    print(f"Epoch {i + 1} | Loss: {loss.item():.6f} | Grad Norm: {grad_norm:.6f}")

print("\nProcessed Correlation Matrices in Training Mode:", output.shape)
print(f"Final Training Loss: {loss.item():.6f}")

# Evaluation mode
liebn.eval()
with th.no_grad():
    test_output = liebn(P)

print("\nProcessed Correlation Matrices in Testing Mode:", test_output.shape)

print("\nTraining and Evaluation completed successfully! 🚀")
