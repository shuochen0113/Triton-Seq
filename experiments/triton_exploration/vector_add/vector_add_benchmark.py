import torch
import triton
import triton.language as tl
import time
import json
import numpy as np

DEVICE = triton.runtime.driver.active.get_current_device()

# Kernel for Single-Block Mode
@triton.jit
def vector_add_single_block(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    num_steps = tl.cdiv(n_elements, BLOCK_SIZE)
    for i in range(num_steps):
        offsets = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        tl.store(output_ptr + offsets, x + y, mask=mask)

# Kernel for Multi-Block Mode
@triton.jit
def vector_add_multi_block(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)

# Helper function to run a kernel
def run_kernel(kernel, x, y, n_elements, BLOCK_SIZE, grid):
    output = torch.empty_like(x)
    torch.cuda.synchronize()
    start_time = time.time()
    kernel[grid](x, y, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    torch.cuda.synchronize()
    elapsed_time = time.time() - start_time
    return output, elapsed_time

# Benchmark function with concise output
def benchmark_vector_add_comparison(sizes, block_sizes):
    results = []

    for size in sizes:
        for BLOCK_SIZE in block_sizes:
            # Use the same input for both tests
            x = torch.randn(size, device=DEVICE)
            y = torch.randn(size, device=DEVICE)

            grid_single = (1,)
            grid_multi = (triton.cdiv(size, BLOCK_SIZE),)

            out_single, time_single = run_kernel(vector_add_single_block, x, y, size, BLOCK_SIZE, grid_single)
            out_multi, time_multi = run_kernel(vector_add_multi_block, x, y, size, BLOCK_SIZE, grid_multi)

            # Convert to numpy for easy comparison
            single_np = out_single.cpu().numpy()
            multi_np = out_multi.cpu().numpy()

            # Compare outputs
            match = np.allclose(single_np, multi_np, atol=1e-6)
            max_diff = float(np.max(np.abs(single_np - multi_np))) if not match else 0.0
            num_mismatched = int(np.sum(np.abs(single_np - multi_np) > 1e-6)) if not match else 0

            results.append({
                "size": size,
                "BLOCK_SIZE": BLOCK_SIZE,
                "Single_Block_Time": round(time_single, 6),
                "Multi_Block_Time": round(time_multi, 6),
                "Outputs_Match": match,
                "Max_Diff": max_diff,
                "Num_Mismatched": num_mismatched
            })

            print(f"Size: {size}, BLOCK_SIZE: {BLOCK_SIZE} -> "
                  f"Single: {time_single:.6f}s, Multi: {time_multi:.6f}s, "
                  f"Match: {match}, Max Diff: {max_diff}")

    return results

# Running Tests
sizes = [2**i for i in range(10, 24)]  # From 1K to ~16M elements
block_sizes = [128, 256, 512]

print("\nRunning Comparison Tests...")
comparison_results = benchmark_vector_add_comparison(sizes, block_sizes)

# Save Results to a Compact JSON File
with open("benchmark_results.json", "w") as f:
    json.dump(comparison_results, f, indent=4)

if all(item["Outputs_Match"] for item in comparison_results):
    print("\nAll outputs match between single-block and multi-block kernels!")
else:
    print("\nMismatch detected! Check 'benchmark_results.json' for details.")

print("\nBenchmark results saved to 'benchmark_results.json'")