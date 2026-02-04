"""
======================================================================================
Project: DSL for Sequence Alignment
File: bench.py
Date: Dec 10, 2025
======================================================================================

Usage:
* Env:
    conda activate /root/emhua/scchen/conda_envs/triton_seq
    
* Compile:
    1. Compile CPU Lib
    g++ -shared -fPIC -O3 -march=native -fopenmp -std=c++17 cpu_sw.cpp -o cpu_sw.so

    2. Compile CUDA Lib (check GPU arch, sm_86 is for A6000, sm_120 is for RTX 5090)
    nvcc -shared -Xcompiler -fPIC -O3 -arch=sm_89 -std=c++20 --extended-lambda --expt-relaxed-constexpr cuda_sw.cu -o cuda_sw.so
    
    3. Compile ThunderKittens Lib (check GPU arch, sm_86 is for A6000, sm_120 is for RTX 5090)
    sed -i 's/static __device__ inline constexpr bf16_2/static __device__ inline bf16_2/g' ThunderKittens/include/common/base_types.cuh
    sed -i 's/static __device__ inline constexpr half_2/static __device__ inline half_2/g' ThunderKittens/include/common/base_types.cuh
    sed -i 's/cuCtxCreate_v4(&contexts\[i\], 0, devices\[i\])/0/g' ThunderKittens/include/types/device/ipc.cuh
    nvcc -shared -Xcompiler -fPIC -O3 -arch=sm_86 -std=c++20 --extended-lambda --expt-relaxed-constexpr -I./ThunderKittens/include tk_sw.cu -o tk_sw.so

    4. Compile Codon Lib
    export LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
    codon build -release --relocation-model=pic -o codon_sw.so codon_sw.codon

* Baseline (CPU) Reference:
    python bench.py --backend cpu --ref datasets/ref.fa --query datasets/query.fa --out results/cpu_result.txt

* Run Others (e.g., CUDA):
    python bench.py --backend cuda --ref datasets/ref.fa --query datasets/query.fa --out results/cuda_result.txt --verify results/cpu_result.txt
    
* One-Click Benchmark (All):
    python bench.py --backend all --ref datasets/ref.fa --query datasets/query.fa

* Environment:
    - Python 3.8+
    - CUDA Toolkit 12.0+ / 13.0+ (for cuTile)
    - A6000 Server (TsinghuaC3I):
        - CPU: Intel(R) Xeon(R) Platinum 8375C CPU @ 2.90GHz
        - GPU: NVIDIA RTX A6000
    - 5090 Server (AutoDL):
        - CPU: Intel(R) Xeon(R) Platinum 8470Q
        - GPU: NVIDIA RTX 5090
"""

import argparse
import time
import ctypes
import numpy as np
import os
import sys
import statistics
import subprocess
from datetime import datetime
from abc import ABC, abstractmethod
import torch

# Ensure Directories Exist
os.makedirs("results", exist_ok=True)
if not os.path.exists("datasets"):
    print("[Warning] 'datasets' directory not found. Please ensure .fa files are placed correctly.")

# ====================================================================================
# System Info Helpers
# ====================================================================================

def get_system_specs():
    """Detects CPU and GPU model names for reporting."""
    cpu_name = "Unknown CPU"
    gpu_name = "Unknown GPU"

    # Detect CPU (Linux)
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if "model name" in line:
                    cpu_name = line.split(":")[1].strip()
                    break
    except Exception:
        pass

    # Detect GPU (nvidia-smi)
    # Method 1: CSV format
    try:
        res = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], 
            text=True
        )
        gpu_name = res.strip().split('\n')[0]
    except Exception as e1:
        # Method 2: Fallback to nvidia-smi -L (more robust in some envs)
        try:
            res = subprocess.check_output(["nvidia-smi", "-L"], text=True)
            # Output example: GPU 0: NVIDIA GeForce RTX 5090 (UUID: GPU-...)
            line = res.strip().split('\n')[0]
            if ":" in line:
                # Parse "NVIDIA GeForce RTX 5090" from "GPU 0: NVIDIA ... (UUID..."
                parts = line.split(':')
                if len(parts) >= 2:
                    gpu_name = parts[1].split('(')[0].strip()
        except Exception as e2:
            print(f"[Warning] GPU detection failed. Error 1: {e1}. Error 2: {e2}")

    return cpu_name, gpu_name

# ====================================================================================
# Data Structures
# ====================================================================================

class AlignResult(ctypes.Structure):
    _fields_ = [
        ("score", ctypes.c_short),
        ("end_ref", ctypes.c_int),
        ("end_query", ctypes.c_int)
    ]

class Dataset:
    def __init__(self, ref_file, query_file):
        print(f"[IO] Loading datasets: {ref_file}, {query_file}")
        self.ref_flat, self.ref_off, self.ref_len = self._load(ref_file)
        self.query_flat, self.query_off, self.query_len = self._load(query_file)
        self.N = min(len(self.ref_off), len(self.query_off))
        if self.N == 0:
            raise ValueError("No sequences found in input files.")
        self._truncate()
        
    def _load(self, filename):
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Dataset not found: {filename}")
        seqs = []
        with open(filename, 'r') as f:
            curr = []
            for line in f:
                line = line.strip()
                if not line: continue
                if line.startswith(">"):
                    if curr: seqs.append("".join(curr))
                    curr = []
                else:
                    curr.append(line)
            if curr: seqs.append("".join(curr))
        return self._pack(seqs)

    def _pack(self, seq_list):
        if not seq_list: return np.array([], dtype=np.int8), np.array([], dtype=np.int32), np.array([], dtype=np.int32)
        table = np.full(256, 4, dtype=np.int8)
        table[ord('A')]=0; table[ord('C')]=1; table[ord('G')]=2; table[ord('T')]=3
        table[ord('a')]=0; table[ord('c')]=1; table[ord('g')]=2; table[ord('t')]=3
        
        encoded = [table[np.frombuffer(s.encode('ascii'), dtype=np.uint8)] for s in seq_list]
        lens = np.array([len(s) for s in encoded], dtype=np.int32)
        offs = np.concatenate(([0], np.cumsum(lens)[:-1])).astype(np.int32) if len(lens)>0 else np.array([], dtype=np.int32)
        flat = np.concatenate(encoded) if encoded else np.array([], dtype=np.int8)
        return flat, offs, lens

    def _truncate(self):
        self.ref_len = self.ref_len[:self.N]
        self.ref_off = self.ref_off[:self.N]
        self.query_len = self.query_len[:self.N]
        self.query_off = self.query_off[:self.N]
        
        self.total_ref_bytes = self.ref_flat.nbytes
        self.total_query_bytes = self.query_flat.nbytes
        self.max_len = int(max(np.max(self.ref_len), np.max(self.query_len))) + 2

# ====================================================================================
# Backend Runners
# ====================================================================================

class BackendRunner(ABC):
    @abstractmethod
    def run(self, dataset: Dataset, threads: int, params: dict) -> list[AlignResult]:
        pass
    
    @property
    @abstractmethod
    def name(self):
        pass

class CTypesBackend(BackendRunner):
    def __init__(self, lib_name, func_name):
        self.lib_path = os.path.join(os.getcwd(), lib_name)
        if not os.path.exists(self.lib_path):
            # Strict check: Error out if .so is missing
            raise FileNotFoundError(f"Shared library '{lib_name}' not found at {self.lib_path}.\n"
                                    f"Please refer to the Usage comments in bench.py to compile it first.")
        try:
            self.lib = ctypes.CDLL(self.lib_path)
            self.func = getattr(self.lib, func_name)
        except OSError as e:
            raise RuntimeError(f"Failed to load {lib_name}: {e}")
            
        self.configure_argtypes()

    @abstractmethod
    def configure_argtypes(self):
        pass

class CPURunner(CTypesBackend):
    def __init__(self):
        super().__init__("cpu_sw.so", "run_sw_cpu")
    
    @property
    def name(self): return "CPU"

    def configure_argtypes(self):
        self.func.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int8), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int8), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(AlignResult),
            ctypes.c_int, # threads
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int # scores
        ]

    def run(self, d, threads, p):
        results = (AlignResult * d.N)()
        self.func(
            d.N,
            d.ref_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            d.ref_off.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            d.ref_len.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            d.query_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            d.query_off.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            d.query_len.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            results,
            threads,
            p['match'], p['mismatch'], p['gap_open'], p['gap_extend']
        )
        return results

class CUDARunner(CTypesBackend):
    def __init__(self):
        super().__init__("cuda_sw.so", "run_sw_cuda")
    
    @property
    def name(self): return "CUDA"

    def configure_argtypes(self):
        self.func.argtypes = [
            ctypes.c_int,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
        ]

    def run(self, d, threads, p):
        import torch
        device = torch.device('cuda')
        t_refs = torch.from_numpy(d.ref_flat).to(device)
        t_ref_off = torch.from_numpy(d.ref_off).to(device)
        t_ref_len = torch.from_numpy(d.ref_len).to(device)
        t_query = torch.from_numpy(d.query_flat).to(device)
        t_query_off = torch.from_numpy(d.query_off).to(device)
        t_query_len = torch.from_numpy(d.query_len).to(device)
        
        t_results = torch.zeros(d.N * 12, dtype=torch.uint8, device=device)
        
        STRIDE = 1
        while STRIDE < d.max_len + 2: STRIDE *= 2
        ws_size = d.N * 3 * STRIDE
        t_H = torch.zeros(ws_size, dtype=torch.int16, device=device)
        t_E = torch.zeros(ws_size, dtype=torch.int16, device=device)
        t_F = torch.zeros(ws_size, dtype=torch.int16, device=device)
        
        self.func(
            d.N,
            t_refs.data_ptr(), t_ref_off.data_ptr(), t_ref_len.data_ptr(),
            t_query.data_ptr(), t_query_off.data_ptr(), t_query_len.data_ptr(),
            t_results.data_ptr(),
            t_H.data_ptr(), t_E.data_ptr(), t_F.data_ptr(),
            STRIDE,
            p['match'], p['mismatch'], p['gap_open'], p['gap_extend']
        )
        
        cpu_buffer = t_results.cpu().numpy().tobytes()
        results = (AlignResult * d.N).from_buffer_copy(cpu_buffer)
        return results

class KittensRunner(CTypesBackend):
    def __init__(self):
        # Purely manual: Check existence only
        super().__init__("tk_sw.so", "run_sw_kittens")
    
    @property
    def name(self): return "ThunderKittens"

    def configure_argtypes(self):
        # Same signature as CUDA
        self.func.argtypes = [
            ctypes.c_int,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
        ]

    def run(self, d, threads, p):
        # Use same Torch Allocator logic as CUDA
        import torch
        device = torch.device('cuda')
        t_refs = torch.from_numpy(d.ref_flat).to(device)
        t_ref_off = torch.from_numpy(d.ref_off).to(device)
        t_ref_len = torch.from_numpy(d.ref_len).to(device)
        t_query = torch.from_numpy(d.query_flat).to(device)
        t_query_off = torch.from_numpy(d.query_off).to(device)
        t_query_len = torch.from_numpy(d.query_len).to(device)
        
        t_results = torch.zeros(d.N * 12, dtype=torch.uint8, device=device)
        
        STRIDE = 1
        while STRIDE < d.max_len + 2: STRIDE *= 2
        ws_size = d.N * 3 * STRIDE
        t_H = torch.zeros(ws_size, dtype=torch.int16, device=device)
        t_E = torch.zeros(ws_size, dtype=torch.int16, device=device)
        t_F = torch.zeros(ws_size, dtype=torch.int16, device=device)
        
        self.func(
            d.N,
            t_refs.data_ptr(), t_ref_off.data_ptr(), t_ref_len.data_ptr(),
            t_query.data_ptr(), t_query_off.data_ptr(), t_query_len.data_ptr(),
            t_results.data_ptr(),
            t_H.data_ptr(), t_E.data_ptr(), t_F.data_ptr(),
            STRIDE,
            p['match'], p['mismatch'], p['gap_open'], p['gap_extend']
        )
        
        cpu_buffer = t_results.cpu().numpy().tobytes()
        results = (AlignResult * d.N).from_buffer_copy(cpu_buffer)
        return results

class TritonRunner(BackendRunner):
    def __init__(self):
        try:
            import triton_sw
            self.module = triton_sw
        except ImportError:
            raise ImportError("Could not import triton_sw.py. Is it in the same directory?")
        except Exception as e:
            raise RuntimeError(f"Triton Error: {e}")
            
    @property
    def name(self): return "Triton"

    def run(self, d, threads, p):
        scores, rs, qs = self.module.run_sw_triton(
            d.N, 
            d.ref_flat, d.ref_off, d.ref_len,
            d.query_flat, d.query_off, d.query_len,
            d.max_len,
            p['match'], p['mismatch'], p['gap_open'], p['gap_extend']
        )
        
        results = (AlignResult * d.N)()
        for i in range(d.N):
            results[i].score = int(scores[i])
            results[i].end_ref = int(rs[i])
            results[i].end_query = int(qs[i])
            
        return results

class TileLangRunner(BackendRunner):
    def __init__(self):
        try:
            sys.path.append(os.getcwd())
            import tilelang_sw
            self.module = tilelang_sw
        except ImportError as e:
            raise ImportError(f"Could not import tilelang_sw.py: {e}. Is TileLang installed?")
        except Exception as e:
            raise RuntimeError(f"TileLang Init Error: {e}")
            
    @property
    def name(self): return "TileLang"

    def run(self, d, threads, p):
        scores, rs, qs = self.module.run_sw_tilelang(
            d.N, 
            d.ref_flat, d.ref_off, d.ref_len,
            d.query_flat, d.query_off, d.query_len,
            d.max_len,
            p['match'], p['mismatch'], p['gap_open'], p['gap_extend']
        )
        
        results = (AlignResult * d.N)()
        for i in range(d.N):
            results[i].score = int(scores[i])
            results[i].end_ref = int(rs[i])
            results[i].end_query = int(qs[i])
            
        return results

class CodonRunner(BackendRunner):
    def __init__(self):
        # We don't verify library existence here because we intentionally skip it
        pass
        
    @property
    def name(self): return "Codon"

    def run(self, d, threads, p):
        print("\n[Warning] Codon GPU support is experimental and lacks necessary primitives for correct S-W implementation.")
        print("          Skipping execution to ensure benchmark integrity.")
        return None

class CuTileRunner(BackendRunner):
    def __init__(self):
        # Strict check for cuTile environment
        try:
            import cuda.tile
        except ImportError as e:
            print("[Error] 'cuda.tile' module not found.")
            raise ImportError("CuTile requires CUDA 13.0+ and a Blackwell Architecture GPU. Please check your environment.") from e
            
        try:
            import cutile_sw
            self.module = cutile_sw
        except ImportError as e:
            raise ImportError(f"Could not import cutile_sw.py: {e}")

    @property
    def name(self): return "CuTile"

    def run(self, d, threads, p):
        scores, rs, qs = self.module.run_sw_cutile(
            d.N, 
            d.ref_flat, d.ref_off, d.ref_len,
            d.query_flat, d.query_off, d.query_len,
            d.max_len,
            p['match'], p['mismatch'], p['gap_open'], p['gap_extend']
        )
        
        results = (AlignResult * d.N)()
        for i in range(d.N):
            results[i].score = int(scores[i])
            results[i].end_ref = int(rs[i])
            results[i].end_query = int(qs[i])
            
        return results

# ====================================================================================
# Benchmark Engine
# ====================================================================================

def verify_results(results, gold_file, N):
    if results is None: return False
    print(f"[Verify] Loading gold standard: {gold_file}")
    gold_data = []
    try:
        with open(gold_file, 'r') as f:
            next(f) # Skip header
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    gold_data.append((int(parts[0]), int(parts[1]), int(parts[2])))
    except Exception as e:
        print(f"[Verify] Error reading gold file: {e}")
        return False
        
    if len(gold_data) != N:
        print(f"[Verify] Error: Length mismatch. Gold={len(gold_data)}, Current={N}")
        return False

    score_mismatches = 0
    coord_mismatches = 0 
    
    for i in range(N):
        gs, gq, gr = gold_data[i]
        cs, cq, cr = results[i].score, results[i].end_query, results[i].end_ref
        
        if cs != gs:
            if score_mismatches < 10: 
                print(f"  [Error] Pair {i}: Score Mismatch. Gold={gs} vs Got={cs}")
            score_mismatches += 1
        elif (cq != gq) or (cr != gr):
            if coord_mismatches < 3:
                print(f"  [Warning] Pair {i}: Coordinate Mismatch. Gold=({gq},{gr}) vs Got=({cq},{cr})")
            coord_mismatches += 1
            
    if score_mismatches == 0:
        msg = "PASS: Perfect Match." if coord_mismatches == 0 else f"PASS (with {coord_mismatches} tie-breaks)."
        print(f"[Verify] {msg}")
        return True
    else:
        print(f"[Verify] FAIL: {score_mismatches} mismatches.")
        return False

def benchmark_single(runner, ds, threads, params, args):
    print(f"[Run] Starting Kernel: {runner.name} ...")
    
    # Warmup
    try:
        runner.run(ds, threads, params)
    except Exception as e:
        print(f"[Error] Execution failed: {e}")
        return

    # Timing
    start_t = time.perf_counter()
    results = runner.run(ds, threads, params)
    end_t = time.perf_counter()

    elapsed = end_t - start_t
    total_cells = np.sum(ds.ref_len.astype(np.float64) * ds.query_len.astype(np.float64))
    gcups = (total_cells / elapsed) / 1e9
    
    print("------------------------------------------------")
    print(f"Time: {elapsed:.4f} s")
    print(f"Perf: {gcups:.2f} GCUPS")
    print("------------------------------------------------")

    if args.out:
        print(f"[IO] Saving to {args.out}")
        with open(args.out, 'w') as f:
            f.write("score\trow\tcol\n")
            if results:
                for r in results:
                    f.write(f"{r.score}\t{r.end_query}\t{r.end_ref}\n")

    if args.verify:
        if not verify_results(results, args.verify, ds.N):
            sys.exit(1)

def benchmark_all(ds, threads, params):
    # Runners to test
    candidates = [
        ('cpu', CPURunner),
        ('cuda', CUDARunner),
        ('triton', TritonRunner),
        ('tilelang', TileLangRunner),
        ('tk', KittensRunner),
        # ('codon', CodonRunner),
        ('cutile', CuTileRunner)
    ]

    # Auto-detect Hardware for Report
    cpu_info, gpu_info = get_system_specs()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Save to history folder with timestamp
    history_dir = os.path.join("results", "bench_history")
    os.makedirs(history_dir, exist_ok=True)
    summary_file = os.path.join(history_dir, f"summary_{timestamp}.txt")
    
    print(f"\n[Auto-Bench] Starting full benchmark suite...")
    print(f"[Auto-Bench] System: CPU={cpu_info}, GPU={gpu_info}")
    print(f"[Auto-Bench] Configuration: N={ds.N}, MaxLen={ds.max_len}")
    print(f"[Auto-Bench] Output will be saved to {summary_file}\n")
    
    results_table = []

    for name, RunnerClass in candidates:
        print(f">>> Benchmarking: {name.upper()}")
        try:
            runner = RunnerClass()
        except Exception as e:
            # Catch initialization errors (missing .so, missing hardware)
            print(f"    [Skip] Init failed: {e}")
            continue
        
        # Explicit skip/check for specific runners if needed
        # Codon logic is handled inside its run() method returning None

        # Config: Repeats
        repeats = 5 if name == 'cpu' else 20
        timings = []
        
        try:
            # Warmup
            print("    Warming up...", end="", flush=True)
            res = runner.run(ds, threads, params)
            if res is None: # Codon returns None
                print(" Skipped.")
                continue
            print(" Done.")

            # Measurement Loop
            print(f"    Running {repeats} iterations...", end="", flush=True)
            for _ in range(repeats):
                t0 = time.perf_counter()
                runner.run(ds, threads, params)
                t1 = time.perf_counter()
                timings.append(t1 - t0)
            print(" Done.")

            # Stats: Remove min/max outlier if enough samples to reduce jitter
            if len(timings) >= 5:
                timings.sort()
                valid_timings = timings[1:-1] # Trim top and bottom
            else:
                valid_timings = timings

            avg_time = statistics.mean(valid_timings)
            total_cells = np.sum(ds.ref_len.astype(np.float64) * ds.query_len.astype(np.float64))
            gcups = (total_cells / avg_time) / 1e9
            
            print(f"    Avg Time: {avg_time:.4f} s | Perf: {gcups:.2f} GCUPS")
            results_table.append((runner.name, avg_time, gcups))

        except Exception as e:
            print(f"\n    [Error] Runtime failure: {e}")

    # Write Summary
    print("\n[Summary] Writing report...")
    with open(summary_file, 'w') as f:
        f.write(f"Benchmark Report - {time.ctime()}\n")
        f.write(f"Hardware: CPU={cpu_info}, GPU={gpu_info}\n")
        f.write(f"Dataset: N={ds.N}, MaxLen={ds.max_len}\n")
        f.write("="*65 + "\n")
        f.write(f"{'Backend':<20} | {'Time (s)':<15} | {'GCUPS':<15}\n")
        f.write("-" * 65 + "\n")
        for name, t, g in results_table:
            line = f"{name:<20} | {t:<15.4f} | {g:<15.2f}"
            print(line)
            f.write(line + "\n")
        f.write("="*65 + "\n")

    print(f"\n[Done] Full benchmark complete. Results in {summary_file}")

# ====================================================================================
# Main
# ====================================================================================

def main():
    parser = argparse.ArgumentParser(description="GPU DSLs for Sequence Alignment Benchmark")
    parser.add_argument("--backend", required=True, choices=['cpu', 'cuda', 'triton', 'codon', 'tilelang', 'tk', 'cutile', 'all'])
    parser.add_argument("--ref", default="datasets/ref.fa")
    parser.add_argument("--query", default="datasets/query.fa")
    parser.add_argument("--out", default=None)
    parser.add_argument("--verify", help="Gold standard file to verify against")
    parser.add_argument("--threads", type=int, default=os.cpu_count())
    
    # Scoring Params
    parser.add_argument("--match", type=int, default=1)
    parser.add_argument("--mismatch", type=int, default=-4)
    parser.add_argument("--gap_open", type=int, default=-6)
    parser.add_argument("--gap_extend", type=int, default=-2)
    
    args = parser.parse_args()

    params = {'match': args.match, 'mismatch': args.mismatch, 'gap_open': args.gap_open, 'gap_extend': args.gap_extend}

    # Load Data
    try:
        ds = Dataset(args.ref, args.query)
    except Exception as e:
        print(f"[Error] Data load failed: {e}")
        return

    # Dispatch
    if args.backend == 'all':
        benchmark_all(ds, args.threads, params)
    else:
        # Single Mode
        try:
            runner = None
            if args.backend == 'cpu': runner = CPURunner()
            elif args.backend == 'cuda': runner = CUDARunner()
            elif args.backend == 'triton': runner = TritonRunner()
            elif args.backend == 'tilelang': runner = TileLangRunner()
            elif args.backend == 'tk': runner = KittensRunner()
            elif args.backend == 'codon': runner = CodonRunner()
            elif args.backend == 'cutile': runner = CuTileRunner()
            
            if runner:
                benchmark_single(runner, ds, args.threads, params, args)
            
        except Exception as e:
            print(f"[Error] {e}")

if __name__ == "__main__":
    main()