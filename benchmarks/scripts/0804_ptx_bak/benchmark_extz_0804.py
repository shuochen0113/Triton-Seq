# benchmarking/scripts/benchmark_extz_0804_fixed.py

"""
A benchmarking script for evaluating the performance of Triton vs Hacked PTX
Usage:
    python benchmark_extz_0804_fixed.py

2025-08-04
"""

import sys
import json
import time
import torch
import os
from pathlib import Path

import cupy as cp
from cupy.cuda import Module

class PtxKernelWrapper:
    """
    一个包装器，用于加载手动修改的PTX文件，并使其能够像Triton JIT函数一样被调用。
    (V7 - 修正kernel启动参数，增加动态共享内存)
    """
    def __init__(self, ptx_file_path: str, kernel_name: str, block_dim: tuple):
        """
        初始化时加载PTX代码。
        """
        print(f"Loading hacked PTX kernel from: {ptx_file_path} using cupy.cuda.Module")
        with open(ptx_file_path, 'r') as f:
            ptx_code = f.read()

        self.module = Module()
        self.module.load(ptx_code.encode('utf-8'))
        self.kernel = self.module.get_function(kernel_name)
        self.block_dim = block_dim
        self.grid_dim = None
        self.__name__ = kernel_name

    def __getitem__(self, grid_dim):
        """
        捕获grid维度，模拟 triton_kernel[grid] 的语法。
        """
        if isinstance(grid_dim, int):
            self.grid_dim = (grid_dim, 1, 1)
        elif isinstance(grid_dim, (tuple, list)):
            if len(grid_dim) == 1: self.grid_dim = (grid_dim[0], 1, 1)
            elif len(grid_dim) == 2: self.grid_dim = (grid_dim[0], grid_dim[1], 1)
            else: self.grid_dim = tuple(grid_dim[:3])
        else:
            raise ValueError(f"Invalid grid dimension type: {type(grid_dim)}")
        return self

    def __call__(self, **kwargs):
        if self.grid_dim is None:
            raise ValueError("Grid dimensions must be set before calling the kernel.")

        # 调试打印
        print(f"Available kwargs keys: {list(kwargs.keys())}")
        
        # 准备参数列表 - 必须按照 PTX 中定义的顺序
        params = [
            # 指针参数 0-7
            kwargs['q_ptrs'].data_ptr(),
            kwargs['r_ptrs'].data_ptr(), 
            kwargs['m_arr'].data_ptr(),
            kwargs['n_arr'].data_ptr(),
            kwargs['outs'].data_ptr(),
            kwargs['Hbuf'].data_ptr(),
            kwargs['Fbuf'].data_ptr(),
            kwargs['Ebuf'].data_ptr(),
            
            # 数值参数 8-11 (必须是 int 类型)
            int(kwargs['match_score']),
            int(kwargs['mismatch_score']),
            int(kwargs['gap_open_penalty']),
            int(kwargs['gap_extend_penalty']),
            
            # param_12 和 param_13 - 看起来是未使用的指针参数
            # 可以传递 0 或者任意有效的指针
            0,  # param_12
            0,  # param_13
        ]
        
        print(f"Total params: {len(params)}")
        print(f"Param types: {[type(p).__name__ for p in params]}")
        
        try:
            # 启动 kernel，提供动态共享内存
            self.kernel(
                grid=self.grid_dim,
                block=self.block_dim, 
                args=tuple(params),
                shared_mem=4096  # 为归约操作提供共享内存
            )
        except Exception as e:
            print(f"Kernel launch failed: {e}")
            print(f"Grid: {self.grid_dim}, Block: {self.block_dim}")
            print(f"Params (count={len(params)}): {params}")
            raise

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

# Import the restored, single API function
from framework_api_OPv3 import run_alignment_task
from configs.task_config import AlignmentTask, SolutionConfiguration, create_affine_scoring, create_banded_z_drop_pruning
from utils.io_OPv3 import read_fasta_as_bytes
from benchmarking.wrappers.ksw2_wrapper import run_ksw2_cli
from benchmarking.wrappers.agatha_wrapper import run_agatha

CONFIG = { 
    "match": 1, 
    "mismatch": -4, 
    "gap_open": -6, 
    "gap_extend": -2, 
    "band_width": 751, 
    "z_drop": 1_000_000_000 
}

def calculate_total_cells(q_seqs, r_seqs):
    total_cells = 0
    for q, r in zip(q_seqs, r_seqs):
        total_cells += len(q) * len(r)
    return total_cells

def main():
    print("--- Starting PTX vs Triton Benchmark ---")

    # Loading data
    load_start_time = time.perf_counter()

    dataset_path = project_root / "datasets"
    try:
        q_seqs = read_fasta_as_bytes(dataset_path / "query_small.fa")
        r_seqs = read_fasta_as_bytes(dataset_path / "ref_small.fa")
        print(f"Loaded {len(q_seqs)} sequence pairs from dataset.")
    except FileNotFoundError:
        print(f"Error: Dataset not found in {dataset_path}.")
        return

    # 在 benchmark 脚本中
    q_seqs = q_seqs[:1]  # 只取第一个
    r_seqs = r_seqs[:1]

    load_end_time = time.perf_counter()
    load_time_ms = (load_end_time - load_start_time) * 1000

    total_cells = calculate_total_cells(q_seqs, r_seqs)
    print(f"Total cells to be computed: {total_cells / 1e9:.3f} GCells")

    results_dir = project_root / "benchmarking" / "results"
    results_dir.mkdir(exist_ok=True)
    
    benchmark_data = {}

    # --- Run Our Framework (Hacked Version) ---
    print("\n[Benchmarking Our HACKED PTX Framework...]")
    our_task = AlignmentTask(
        align_problem='Extension',
        configuration=SolutionConfiguration(
            scoring=create_affine_scoring(CONFIG['match'], CONFIG['mismatch'], CONFIG['gap_open'], CONFIG['gap_extend']),
            pruning=create_banded_z_drop_pruning(CONFIG['band_width'], CONFIG['z_drop'])
        )
    )

    # 手动创建scheduler而不是通过run_alignment_task
    from core.local_dp_kernel_OPv3 import sw_kernel
    from host.scheduler_OPv3 import PipelineScheduler

    band_width = CONFIG['band_width']
    STRIDE = 2 * band_width + 1
    block_size = 256  # 根据PTX文件中的.reqntid 128
    
    kernel_args_template = {
        'match_score': CONFIG['match'], 
        'mismatch_score': CONFIG['mismatch'],
        'gap_open_penalty': CONFIG['gap_open'], 
        'gap_extend_penalty': CONFIG['gap_extend'],
        'drop_threshold': CONFIG['z_drop'], 
        'BLOCK': block_size,
        'SCORING_MODEL': 'AFFINE', 
        'PRUNING_BAND': 'STATIC',
        'PRUNING_DROP': 'Z_DROP', 
        'IS_EXTENSION': (our_task.align_problem == 'Extension'),
        'STRIDE': STRIDE, 
        'BAND': band_width
    }

    # 创建Scheduler实例
    scheduler = PipelineScheduler(
        kernel_to_run=sw_kernel,  # 先用原始kernel占位
        kernel_args_template=kernel_args_template,
        cfg=our_task.configuration,
        batch_size=65536,  # 减小batch size以便调试
        max_seq_len=10000,
        n_streams=1  # 先用单stream调试
    )
    
    # 注入我们修改后的PTX Kernel！
    ptx_kernel_path = project_root / "benchmarking" / "scripts" / "smem2.ptx"
    if not ptx_kernel_path.exists():
        print(f"PTX file not found: {ptx_kernel_path}")
        return
        
    scheduler.kernel_to_run = PtxKernelWrapper(
        ptx_file_path=str(ptx_kernel_path), 
        kernel_name='sw_kernel', 
        block_dim=(block_size, 1, 1)
    )
    
    print("Performing timed execution run with HACKED PTX KERNEL...")
    
    # 执行前先同步确保clean state
    torch.cuda.synchronize()
    
    start_time = time.perf_counter()
    
    try:
        # 执行调度器
        all_results, total_kernel_time_ms, total_setup_ms, packing_kernel_time_ms, total_d2h_time_ms = scheduler.execute(q_seqs, r_seqs)
        
        end_time = time.perf_counter()
        our_process_time_ms = (end_time - start_time) * 1000

        # 计算总GPU工作时间
        total_gpu_work_time = total_setup_ms + total_kernel_time_ms + total_d2h_time_ms
        
        # 计算pipeline节省的时间
        pipeline_overlap_ms = max(0, total_gpu_work_time - our_process_time_ms)

        print("\n--- Hacked PTX Framework Performance Breakdown ---")
        print(f"Total Wall-Clock Time: {our_process_time_ms + load_time_ms:.3f} ms")
        print(f"Load Time: {load_time_ms:.3f} ms")
        print(f"Process Time: {our_process_time_ms:.3f} ms")
        print("-" * 50)
        print("Accumulated GPU Task Times:")
        print(f"  - Alignment Kernel Time        : {total_kernel_time_ms:.3f} ms")
        print(f"  - Setup Time (H2D, Pack, Init) : {total_setup_ms:.3f} ms")
        print(f"      (-> Packing Kernel Time)   : {packing_kernel_time_ms:.3f} ms")
        print(f"  - D2H Transfer Time            : {total_d2h_time_ms:.3f} ms")
        print(f"  -------------------------------------------------")
        print(f"  = Total GPU Work Scheduled     : {total_gpu_work_time:.3f} ms")
        print("-" * 50)
        print("Pipeline Efficiency:")
        if our_process_time_ms > 0:
            print(f"  => Time Saved by Overlap        : {pipeline_overlap_ms:.3f} ms ({pipeline_overlap_ms / total_gpu_work_time * 100:.2f}% of total work)")
        print("-" * 50)    
        
        benchmark_data['Hacked PTX Framework'] = {
            'kernel_ms': total_kernel_time_ms, 
            'total_ms': our_process_time_ms
        }

        # Save results
        with open(results_dir / "hacked_ptx_results.json", 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"Results for {len(all_results)} pairs saved.")
        
    except Exception as e:
        print(f"ERROR during PTX kernel execution: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- 对比：运行原始Triton版本 ---
    print("\n[Benchmarking Original Triton Framework for comparison...]")
    
    try:
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        triton_results, triton_kernel_time_ms, triton_setup_ms, triton_packing_ms, triton_d2h_ms = run_alignment_task(q_seqs, r_seqs, our_task)
        
        end_time = time.perf_counter()
        triton_process_time_ms = (end_time - start_time) * 1000
        
        print(f"Original Triton Process Time: {triton_process_time_ms:.3f} ms")
        print(f"Original Triton Kernel Time: {triton_kernel_time_ms:.3f} ms")
        
        benchmark_data['Original Triton'] = {
            'kernel_ms': triton_kernel_time_ms,
            'total_ms': triton_process_time_ms
        }
        
        # 保存Triton结果用于对比
        with open(results_dir / "triton_results.json", 'w') as f:
            json.dump(triton_results, f, indent=2)
            
    except Exception as e:
        print(f"ERROR during Triton execution: {e}")
        import traceback
        traceback.print_exc()

    # --- Print Final Summary Table ---
    print("\n--- Benchmark Summary ---")
    print(f"{'Framework':<25} | {'Kernel Time (ms)':<18} | {'Total Runtime (ms)':<20} | {'Throughput (GCUPS)':<20}")
    print("-" * 90)
    
    for tool, data in benchmark_data.items():
        kernel_t = f"{data['kernel_ms']:.3f}" if data['kernel_ms'] is not None and data['kernel_ms'] > 0 else "N/A"
        total_t = f"{data['total_ms']:.3f}"
        gcups = f"{(total_cells / (data['total_ms'] / 1000)) / 1e9:.3f}" if data['total_ms'] > 0 else "N/A"
        
        print(f"{tool:<25} | {kernel_t:<18} | {total_t:<20} | {gcups:<20}")

    print("\n--- Analysis Complete ---")

# --- Script Entry Point ---
if __name__ == "__main__":
    main()