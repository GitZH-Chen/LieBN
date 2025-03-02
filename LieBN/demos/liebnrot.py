"""
Demo for LieBN on SO(3) matrices

@author: Ziheng Chen
Please cite:

Ziheng Chen, Yue Song, Yunmei Liu, Nicu Sebe.
A Lie Group Approach to Riemannian Batch Normalization. ICLR 2024

Copyright (C) 2024 Ziheng Chen
All rights reserved.
"""

import torch as th
import geoopt
from LieBN import LieBNRot
from LieBN.Geometry.Rotations import RotMatrices

# Set the random seed for reproducibility
SEED = 42
th.manual_seed(SEED)

# Batch size, frame, space (SO(3) are 3x3 matrices)
# In LieNet, the shape is [bs,f,s,3,3]
bs, f, s = 5, 2, 3
manifold = RotMatrices()

# Generate random SO(3) matrices as inputs and targets
input_rot = manifold.random(bs,f, s,3,3).to(th.double).requires_grad_(True).to(th.double)  # Input SO(3) matrices
target_rot = manifold.random(bs,f, s,3,3).to(th.double).to(th.double)  # Target matrices for loss computation

# Instantiate LieBN for rotation matrices
# is_left=False/True
# liebn = LieBNRot(shape=[s,3,3], batchdim=[0, 1], is_left=False,karcher_steps=100).to(th.double)
liebn = LieBNRot(shape=[s,3,3], batchdim=[0,1], is_left=False).to(th.double)

print("\nLieBNRot Layer Initialized:", liebn)
print("Manifold:", liebn.manifold)

print("\n=== LieBNRot Parameters ===")
for name, param in liebn.named_parameters():
    print(f"Parameter Name: {name}, Shape: {param.shape}, Requires Grad: {param.requires_grad}")

# Define optimizer and loss function
optimizer = geoopt.optim.RiemannianAdam(liebn.parameters(), lr=1e-3)
criterion = th.nn.MSELoss()

# Training loop
num_epochs = 2  # Set higher if needed
for i in range(num_epochs):
    liebn.train()  # Set to training mode
    optimizer.zero_grad()  # Clear previous gradients

    # Forward pass
    output_rot = liebn(input_rot)

    # Compute loss
    loss = criterion(output_rot, target_rot)

    # Backpropagation
    loss.backward()
    optimizer.step()

    # Print loss every iteration
    print(f"Epoch {i + 1} | Loss: {loss.item():.6f}")

print("\nProcessed SO(3) Matrices in Training Mode:", output_rot.shape)
print(f"Final Training Loss: {loss.item():.6f}")

# Evaluation mode
liebn.eval()
with th.no_grad():
    test_output = liebn(input_rot)

print("\nProcessed SO(3) Matrices in Testing Mode:", test_output.shape)

print("\nTraining and Evaluation completed successfully! 🚀")
