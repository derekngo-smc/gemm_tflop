import torch
import torch.distributed as dist
import os

# Initialize the distributed environment.
dist.init_process_group(backend='nccl')

# Set the device to use all 8 GPUs.
device = torch.device(f'cuda:{dist.get_rank()}')

# Create the matrices with the desired dimensions and dtype.
#matrix_size = 102400
matrix_size = 102400
A = torch.randn(matrix_size, matrix_size, device=device, dtype=torch.bfloat16)
B = torch.randn(matrix_size, matrix_size, device=device, dtype=torch.bfloat16)

# Function to perform the matrix multiplication.
def matmul_forever():
    while True:
        C = torch.matmul(A, B)

    torch.cuda.synchronize()  # Ensure all GPU operations are completed.

if __name__ == "__main__":
    # Run the matmul function forever.
    matmul_forever()
