"""
Demo for LieBN on SPD Matrices

@author: Ziheng Chen
Please cite:

Ziheng Chen, Yue Song, Yunmei Liu, Nicu Sebe.
A Lie Group Approach to Riemannian Batch Normalization. ICLR 2024

Copyright (C) 2024 Ziheng Chen
All rights reserved.
"""

import torch as th
import geoopt
from LieBN import LieBNSPD
from LieBN.Geometry.SPD import SPDMatrices

# Set the random seed for reproducibility
SEED = 42
th.manual_seed(SEED)


# Batch size, channels, SPD matrix size
# In SPDNet and SPDNetBN, the shape is [bs,1,n,n]
bs, c, n = 4, 2, 5

# Generate input SPD matrices and target SPD matrices
manifold=SPDMatrices(n=n)
P = manifold.random(bs, c, n,n).requires_grad_(True).to(th.double)  # Input SPD matrices
target = manifold.random(bs, c, n,n).to(th.double)  # Target SPD matrices for loss computation

# LEM,ALEM,LCM,AIM,CRIM
liebn = LieBNSPD(shape=[c, n, n], metric="LCM", batchdim=[0]).to(th.double)

print("\nLieBNSPD Layer Initialized:", liebn)

print("\n=== LieBNSPD Parameters ===")
for name, param in liebn.named_parameters():
    print(f"Parameter Name: {name}, Shape: {param.shape}, Requires Grad: {param.requires_grad}")

# Define optimizer (Adam) and loss function (MSE)
optimizer = geoopt.optim.RiemannianAdam(liebn.parameters(), lr=1e-3)
criterion = th.nn.MSELoss()

# Print initial loss
with th.no_grad():
    initial_loss = criterion(liebn(P), target).item()
print(f"\nInitial Loss: {initial_loss:.6f}")

# Training loop
num_epochs = 2  # Increase if needed
for i in range(num_epochs):
    liebn.train()  # Set to training mode
    optimizer.zero_grad()  # Clear previous gradients

    # Forward pass
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

print("\nProcessed SPD Matrices in Training Mode:", output.shape)
print(f"Final Training Loss: {loss.item():.6f}")

# Evaluation mode
liebn.eval()
with th.no_grad():
    test_output = liebn(P)

print("\nProcessed SPD Matrices in Testing Mode:", test_output.shape)

print("\nTraining and Evaluation completed successfully! 🚀")
