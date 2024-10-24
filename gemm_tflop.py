import torch
import os
import time


def run_gemm_bf16():

    device = torch.device(f'cuda:{0}')

    M, N, K = 8192, 8192, 8192

    A = torch.randn(M, K, device=device, dtype=torch.bfloat16)
    B = torch.randn(K, N, device=device, dtype=torch.bfloat16)

    torch.matmul(A, B)
    torch.cuda.synchronize()

    num_runs = 2000
    start_time = time.perf_counter()
    for _ in range(num_runs):
        C = torch.matmul(A, B)
    torch.cuda.synchronize()
    end_time = time.perf_counter()

    elapsed_time = end_time - start_time
    flops = 2 * M * N * K * num_runs  # multiply-add is 2 operations
    tflops = flops / (elapsed_time * 1e12)
    print(f"Realized TFLOPS: {tflops:.2f}")

if __name__ == "__main__":
    run_gemm_bf16()
