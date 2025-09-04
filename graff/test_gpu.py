#!/usr/bin/env python3

import torch
import os

print("=== GPU Test ===")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"CUDA device count: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
    
    # Test GPU usage
    print("\n=== Testing GPU Usage ===")
    device = torch.device("cuda:0")
    print(f"Using device: {device}")
    
    # Create a tensor on GPU
    x = torch.randn(1000, 1000, device=device)
    y = torch.randn(1000, 1000, device=device)
    
    # Perform computation
    z = torch.matmul(x, y)
    print(f"Matrix multiplication result shape: {z.shape}")
    print(f"Result device: {z.device}")
    
    # Check memory usage
    print(f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")
    print(f"GPU memory cached: {torch.cuda.memory_reserved(0) / 1024**2:.1f} MB")
    
else:
    print("CUDA is not available!")

print("\n=== Environment Variables ===")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
