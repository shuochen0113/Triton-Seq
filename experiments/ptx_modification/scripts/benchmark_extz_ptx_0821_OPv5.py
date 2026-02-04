# benchmarking/scripts/benchmark_extz_ptx_0810.py
"""
PTX-based benchmark for the SW (banded + Z-drop, affine/linear) wavefront kernel.

Goal: Load the hand-edited PTX, launch it via CUDA Driver API, and
compare correctness & performance against the existing Triton baseline
(with minimal changes elsewhere).

Usage:
    python benchmark_extz_ptx_0810.py

Notes:
- Only this script is modified. BufferManager / Scheduler / framework_api remain intact.
- We run a tiny dataset (first 3 pairs) to validate correctness and timing.
- Dynamic shared memory is required by the PTX version. We compute it from BAND (STRIDE).
"""

import sys
import json
import time
import csv
import ctypes
from pathlib import Path

import torch
import numpy as np
from datetime import datetime

# Project setup
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

# Reuse existing utilities / configs / baseline API
from utils.io_OPv5 import read_fasta_as_bytes
from utils.packing_kernel_OPv5 import pack_batch_into_buffers
from host.buffer_manager_OPv4 import create_pipeline_resources
from configs.task_config import AlignmentTask, SolutionConfiguration, create_affine_scoring, create_banded_z_drop_pruning
from framework_api_OPv5 import run_alignment_task

# Prefer the official CUDA Python bindings
# try:
#     from cudaimport cuda
# except Exception as e:  # pragma: no cover
#     raise RuntimeError("This benchmark requires the 'cuda-python' package. Please install nvidia-cuda-python.") from e

try:
    # Preferred: nvidia-cuda-python high-level wrapper
    from cuda import cuda  # normal route
    _CUDA_MOD_NAME = "cuda"
except Exception:
    # Fallback: low-level Cython bindings
    from cuda.bindings import driver as cuda
    _CUDA_MOD_NAME = "cuda.bindings.driver"

# ---- enum/attr shim for cuda.bindings.driver (it misses many constants) ----
# Provide numeric fallbacks per CUDA Driver API enums
if not hasattr(cuda, "CU_JIT_MAX_REGISTERS"):
    # CUjit_option
    cuda.CU_JIT_MAX_REGISTERS                 = 0
    cuda.CU_JIT_THREADS_PER_BLOCK             = 1
    cuda.CU_JIT_WALL_TIME                     = 2
    cuda.CU_JIT_INFO_LOG_BUFFER               = 3
    cuda.CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES    = 4
    cuda.CU_JIT_ERROR_LOG_BUFFER              = 5
    cuda.CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES   = 6
    cuda.CU_JIT_OPTIMIZATION_LEVEL            = 7
    cuda.CU_JIT_TARGET_FROM_CUCONTEXT         = 8
    cuda.CU_JIT_TARGET                        = 9
    cuda.CU_JIT_FALLBACK_STRATEGY             = 10
    cuda.CU_JIT_GENERATE_DEBUG_INFO           = 11
    cuda.CU_JIT_LOG_VERBOSE                   = 12
    cuda.CU_JIT_GENERATE_LINE_INFO            = 13
    # (rest omitted; not used here)

if not hasattr(cuda, "CU_FUNC_ATTRIBUTE_NUM_REGS"):
    # CUfunction_attribute
    cuda.CU_FUNC_ATTRIBUTE_NUM_REGS                         = 0
    cuda.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES                = 1
    cuda.CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES                 = 2
    cuda.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES                 = 3
    cuda.CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK            = 4
    cuda.CU_FUNC_ATTRIBUTE_PTX_VERSION                      = 5
    cuda.CU_FUNC_ATTRIBUTE_BINARY_VERSION                   = 6
    cuda.CU_FUNC_ATTRIBUTE_CACHE_MODE_CA                    = 7  # deprecated but harmless
    cuda.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES    = 8
    cuda.CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT = 9

# helper functions to access these enums
def _CUFUNC(name: str, fallback_int: int):
    if hasattr(cuda, "CUfunction_attribute"):
        return getattr(cuda.CUfunction_attribute, name)
    if _is_bindings_driver():
        return cuda.CUfunction_attribute(fallback_int)
    return fallback_int

def _CUJIT(name: str, fallback_int: int):
    if hasattr(cuda, "CUjit_option"):
        return getattr(cuda.CUjit_option, name)
    if _is_bindings_driver():
        return cuda.CUjit_option(fallback_int)
    return fallback_int

if not hasattr(cuda, "CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT"):
    # CUdevice_attribute (we only need SM count here)
    cuda.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16

# --- Backend helpers: unify enum/option types across cuda vs cuda.bindings.driver ---
def _is_bindings_driver():
    # True when we imported 'from cuda.bindings import driver as cuda'
    return _CUDA_MOD_NAME != "cuda"

def _jit_option(x):
    # Wrap int -> CUjit_option for low-level bindings; no-op for high-level
    try:
        return cuda.CUjit_option(x) if _is_bindings_driver() else x
    except Exception:
        return x

def _func_attr(x):
    # Wrap int -> CUfunction_attribute for low-level bindings
    try:
        return cuda.CUfunction_attribute(x) if _is_bindings_driver() else x
    except Exception:
        return x

def _device_attr(x):
    # Wrap int -> CUdevice_attribute for low-level bindings
    try:
        return cuda.CUdevice_attribute(x) if _is_bindings_driver() else x
    except Exception:
        return x

# Helper to parse .reqntid from PTX file
def parse_reqntid(ptx_path: Path):
    try:
        txt = ptx_path.read_text(errors="ignore")
    except Exception:
        return ""
    for line in txt.splitlines():
        line = line.strip()
        if line.startswith(".reqntid"):
            return line  # e.g. ".reqntid 128, 1, 1"
    return ""

# ===================== Config =====================
CONFIG = {"match": 1, "mismatch": -4, "gap_open": -6, "gap_extend": -2, "band_width": 751, "z_drop": 400}
PTX_PATH = project_root / "ptx_modification_0804" / "ptx" / "OPv5_old" / "sw_kernel_OPv5_hack_H.ptx"
TRITON_PTX_PATH = project_root / "ptx_modification_0804" / "ptx" /"OPv5_old" / "sw_kernel_OPv5.ptx"
FUNC_NAME = b"sw_kernel"  # PTX .entry name
NUM_PAIRS = 65536

# ==================================================
# Helpers
# ==================================================

def ensure_cuda_context():
    cuda.cuInit(0)
    _res, ctx = cuda.cuCtxGetCurrent()
    if ctx is None or ctx == 0:
        dev = cuda.cuDeviceGet(0)[1]
        ctx = cuda.cuCtxCreate(0, dev)[1]
    return ctx


# def load_ptx_function(ptx_path: Path, func_name: bytes):
#     if not ptx_path.exists():
#         raise FileNotFoundError(f"PTX not found: {ptx_path}")
#     with open(ptx_path, "rb") as f:
#         ptx = f.read()
#     mod = cuda.cuModuleLoadData(ptx)[1]
#     fn = cuda.cuModuleGetFunction(mod, func_name)[1]
#     return mod, fn

def load_ptx_function(ptx_path: Path, func_name: bytes, *, maxrreg: int = 0, carveout: int = -1):
    if not ptx_path.exists():
        raise FileNotFoundError(f"PTX not found: {ptx_path}")
    ptx = ptx_path.read_bytes()

    info_log = ctypes.create_string_buffer(8192)
    used_ex = False
    mod = None

    # Preferred: high-level cuda-python Ex path (stable types)
    if _CUDA_MOD_NAME == "cuda" and hasattr(cuda, "cuModuleLoadDataEx"):
        try:
            opt_keys = (ctypes.c_int * 4)(
                cuda.CU_JIT_INFO_LOG_BUFFER,
                cuda.CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
                cuda.CU_JIT_LOG_VERBOSE,
                cuda.CU_JIT_MAX_REGISTERS,
            )
            opt_vals = (ctypes.c_void_p * 4)(
                ctypes.c_void_p(ctypes.addressof(info_log)),
                ctypes.c_void_p(ctypes.sizeof(info_log)),
                ctypes.c_void_p(1),
                ctypes.c_void_p(int(maxrreg) if maxrreg > 0 else 0),
            )
            mod = cuda.cuModuleLoadDataEx(ptx, 4, opt_keys, opt_vals)[1]
            used_ex = True
        except Exception as e:
            # Any enum/type mismatch -> fall back to the simplest API
            print(f"[PTX JIT] cuModuleLoadDataEx failed on high-level path ({type(e).__name__}: {e}); falling back to cuModuleLoadData...")
            mod = None

    if mod is None:
        # Fallback: simplest API with no options — works on both the high-level and low-level driver bindings.
        mod = cuda.cuModuleLoadData(ptx)[1]

    fn = cuda.cuModuleGetFunction(mod, func_name)[1]

    # Optional carveout; ignore if not supported on this backend
    try:
        if carveout >= 0:
            cuda.cuFuncSetAttribute(fn, _func_attr(cuda.CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT), int(carveout))
    except Exception:
        pass

    # Query and print function attributes when available
    try:
        regs = cuda.cuFuncGetAttribute(_CUFUNC("CU_FUNC_ATTRIBUTE_NUM_REGS", 4), fn)[1]
        sh_static = cuda.cuFuncGetAttribute(_CUFUNC("CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES", 1), fn)[1]
        local_bytes = cuda.cuFuncGetAttribute(_CUFUNC("CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES", 3), fn)[1]
        max_threads = cuda.cuFuncGetAttribute(_CUFUNC("CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK", 0), fn)[1]
        print(f"[PTX JIT] regs/thread={regs}, shared_static={sh_static}B, local_bytes/thread={local_bytes}, max_threads/block={max_threads}")        
        if used_ex:
            print(f"[PTX JIT log]\n{info_log.value.decode(errors='ignore')}")
        else:
            print("[PTX JIT] (no Ex options used; loaded via cuModuleLoadData)")
    except Exception:
        pass
    
    return mod, fn

def diag_band_cells(m: int, n: int, band: int) -> int:
    """Exact cell count for STATIC band (used for GCUPS)."""
    cells = 0
    for d in range(2, m + n + 1):
        i_min = max(1, d - n)
        i_max = min(m, d - 1)
        half_lo = (d - band + 1) >> 1
        half_hi = (d + band) >> 1
        i_lo = max(i_min, half_lo)
        i_hi = min(i_max, half_hi)
        if i_hi >= i_lo:
            cells += (i_hi - i_lo + 1)
    return cells


def gcups_from_cells(cells: int, ms: float) -> float:
    return (cells / 1e9) / (ms / 1e3) if ms > 0 else 0.0

def analyze_and_save_mismatches(ptx_outs: np.ndarray,
                                base_arr: np.ndarray,
                                q_list,
                                r_list,
                                results_dir: Path,
                                tag: str = "") -> str:
    """
    Save only mismatches (not all results) into a separate JSON file.
    Returns the saved filepath (str). Also prints a concise console summary.
    """
    assert ptx_outs.shape == base_arr.shape, "Shape mismatch in outputs."
    # Row-wise mismatch
    row_mismatch = np.any(ptx_outs != base_arr, axis=1)
    total = int(row_mismatch.sum())

    # Categorize
    score_only_mask = row_mismatch & \
                      (ptx_outs[:, 1] == base_arr[:, 1]) & \
                      (ptx_outs[:, 2] == base_arr[:, 2]) & \
                      (ptx_outs[:, 0] != base_arr[:, 0])
    coord_mask = row_mismatch & ~score_only_mask

    score_only_idx = np.where(score_only_mask)[0].tolist()
    coord_idx = np.where(coord_mask)[0].tolist()

    # Build compact details (only save mismatch)
    details = []
    for idx in np.where(row_mismatch)[0].tolist():
        details.append({
            "index": int(idx),
            "ptx": ptx_outs[idx].tolist(),
            "baseline": base_arr[idx].tolist(),
            "diffs": {
                "score": bool(int(ptx_outs[idx, 0] != base_arr[idx, 0])),
                "i":     bool(int(ptx_outs[idx, 1] != base_arr[idx, 1])),
                "j":     bool(int(ptx_outs[idx, 2] != base_arr[idx, 2])),
            },
            "lens": {"m": int(len(q_list[idx])), "n": int(len(r_list[idx]))},
        })

    payload = {
        "summary": {
            "total_pairs": int(len(q_list)),
            "mismatch_pairs": total,
            "score_only_mismatch": int(len(score_only_idx)),
            "coord_mismatch": int(len(coord_idx)),
            # index lists are capped at 2000 each
            "score_only_indices": score_only_idx[:2000],
            "coord_indices": coord_idx[:2000],
        },
        "mismatches": details
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = (tag or f"np{len(q_list)}")
    fname = f"ptx_mismatches_{tag}_{total}_{len(q_list)}_{ts}.json"
    out_path = results_dir / fname
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"[MISMATCH] {total} / {len(q_list)} pairs differ "
          f"(score-only={len(score_only_idx)}, coords={len(coord_idx)}). "
          f"Details: {out_path}")
    return str(out_path)


# ==================================================
# PTX Kernel Runner
# ==================================================

def run_ptx_alignment(q_list, r_list, *, ptx_path=PTX_PATH, use_shared: bool = True,
                      label: str = "[PTX Shared ]", carveout: int = 100,
                      force_equal_smem: bool = False):
    """Launch the PTX kernel and return (outs_cpu, kernel_ms, cells_static, metrics_dict)."""
    assert len(q_list) == len(r_list)
    batch_size = len(q_list)
    max_seq_len = max(max(len(q) for q in q_list), max(len(r) for r in r_list))

    # Prepare config / STRIDE
    band_width = CONFIG['band_width']
    STRIDE = 2 * band_width + 1
    # STRIDE = band_width

    # 1) Reuse existing buffer creation to minimize changes
    host_bufs, dev_bufs = create_pipeline_resources(
        kernel_name='sw_kernel',
        cfg=SolutionConfiguration(
            scoring=create_affine_scoring(CONFIG['match'], CONFIG['mismatch'], CONFIG['gap_open'], CONFIG['gap_extend']),
            pruning=create_banded_z_drop_pruning(CONFIG['band_width'], CONFIG['z_drop'])
        ),
        n_streams=1,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        device='cuda'
    )

    # 2) Pack sequences using our existing packer (keeps encoding consistent)
    pack_events = {}
    pack_batch_into_buffers(q_list, dev_bufs[0]['q'], pack_events)
    pack_events = {}
    pack_batch_into_buffers(r_list, dev_bufs[0]['r'], pack_events)

    # 3) Build q_ptrs / r_ptrs device arrays (int64 addresses)
    q_base = dev_bufs[0]['q']['packed_u32'].data_ptr()
    r_base = dev_bufs[0]['r']['packed_u32'].data_ptr()
    q_off = dev_bufs[0]['q']['offsets'][:batch_size].to(torch.int64)
    r_off = dev_bufs[0]['r']['offsets'][:batch_size].to(torch.int64)
    q_ptrs_arr = (q_base + q_off * 4).to(torch.int64)
    r_ptrs_arr = (r_base + r_off * 4).to(torch.int64)

    # Ensure they are materialized as device arrays
    q_ptrs_dev = q_ptrs_arr.contiguous()
    r_ptrs_dev = r_ptrs_arr.contiguous()

    # 4) Pointers for all params
    d_m = dev_bufs[0]['q']['lengths'][:batch_size].contiguous()
    d_n = dev_bufs[0]['r']['lengths'][:batch_size].contiguous()
    d_outs = dev_bufs[0]['outs'][:batch_size]

    d_H = dev_bufs[0]['dp']['Hbuf'][:batch_size]
    d_F = dev_bufs[0]['dp']['Fbuf'][:batch_size]
    d_E = dev_bufs[0]['dp']['Ebuf'][:batch_size]

    # 5) Dynamic shared memory size
    header_bytes = 64
    # ===== CORRECTED: STRIDE IS BAND_WIDTH, NOT 2*BAND_WIDTH+1 =====
    # STRIDE = band_width
    STRIDE = 2 * band_width + 1
    # header(64) + H:3*STRIDE*4 + E:2*STRIDE*4 + F:2*STRIDE*4  => 64 + 7*STRIDE*4
    # ===== MODIFY THIS FOR DIFFERENT H/E/F BUFFER SIZES =====
    # shared_bytes_formula = ((header_bytes + 7 * STRIDE * 4 + 15) // 16) * 16
    shared_bytes_formula = ((header_bytes + 7 * STRIDE * 2 + 15) // 16) * 16

    if use_shared or force_equal_smem:
        shared_bytes = shared_bytes_formula
    else:
        shared_bytes = 0

    # 6) Load PTX function
    ensure_cuda_context()
    mod, fn = load_ptx_function(ptx_path, FUNC_NAME, maxrreg=0, carveout=(carveout if use_shared else -1))

    # if needed, declare max dynamic shared (only effective for use_shared)
    if use_shared and shared_bytes > 0:
        try:
            cuda.cuFuncSetAttribute(
                fn,
                _func_attr(cuda.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES),
                int(shared_bytes)
            )
        except Exception as e:
            print(f"[Warn] {label} set MAX_DYNAMIC_SHARED_SIZE_BYTES failed:", e)

    # Query function attributes for metrics
    regs = sh_static = local_bytes = max_threads = None
    try:
        regs = cuda.cuFuncGetAttribute(_CUFUNC("CU_FUNC_ATTRIBUTE_NUM_REGS", 4), fn)[1]
        sh_static = cuda.cuFuncGetAttribute(_CUFUNC("CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES", 1), fn)[1]
        local_bytes = cuda.cuFuncGetAttribute(_CUFUNC("CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES", 3), fn)[1]
        max_threads = cuda.cuFuncGetAttribute(_CUFUNC("CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK", 0), fn)[1]
    except Exception:
        pass

    # 7) Prepare kernel arguments (ensure order matches .entry definition)
    args_np = [
        np.array([q_ptrs_dev.data_ptr()], dtype=np.uint64),
        np.array([r_ptrs_dev.data_ptr()], dtype=np.uint64),
        np.array([d_m.data_ptr()], dtype=np.uint64),
        np.array([d_n.data_ptr()], dtype=np.uint64),
        np.array([d_outs.data_ptr()], dtype=np.uint64),
        np.array([d_H.data_ptr()], dtype=np.uint64),
        np.array([d_F.data_ptr()], dtype=np.uint64),
        np.array([d_E.data_ptr()], dtype=np.uint64),
        np.array([CONFIG['mismatch']], dtype=np.int32),
        np.array([CONFIG['gap_open']], dtype=np.int32),
        np.array([CONFIG['gap_extend']], dtype=np.int32),
        np.array([CONFIG['z_drop']], dtype=np.int32),
        np.array([d_outs.data_ptr()], dtype=np.uint64),  # dummy for param 12
        np.array([d_outs.data_ptr()], dtype=np.uint64),  # dummy for param 13
    ]
    kernelParams = (ctypes.c_void_p * len(args_np))(
        *[ctypes.c_void_p(int(a.ctypes.data)) for a in args_np]
    )
    kernelParams_ptr = int(ctypes.addressof(kernelParams))

    # 8) Launch
    block_x = 128  # must match .reqntid 128
    grid_x = batch_size

    # Occupancy
    active_blocks = None
    sms = None
    try:
        # Some backends return (res, val) with val possibly None; coerce carefully.
        _res, _val = cuda.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn, block_x, int(shared_bytes))
        active_blocks = int(_val) if _val is not None else None
        try:
            dev = cuda.cuDeviceGet(0)[1]
            sms = cuda.cuDeviceGetAttribute(_device_attr(cuda.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT), dev)[1]
        except Exception:
            try:
                sms = torch.cuda.get_device_properties(0).multi_processor_count
            except Exception:
                sms = None
        if (active_blocks is not None) and (sms is not None):
            print(f"[Occu {label}] active_blocks/SM={active_blocks}, total_blocks_theoretical={active_blocks * int(sms)}")
        else:
            print(f"[Occu {label}] active_blocks/SM={active_blocks}, total_blocks_theoretical=N/A")
    except Exception as e:
        print(f"[Occu {label}] (skip) {e}")

    smem_per_blk_optin = None
    try:
        dev = cuda.cuDeviceGet(0)[1]
        _attr_id = __import__('builtins').getattr(cuda, 'CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN', 97)
        smem_per_blk_optin = cuda.cuDeviceGetAttribute(_device_attr(_attr_id), dev)[1]
        print(f"[Device {label}] max_dyn_smem/block_optin={int(smem_per_blk_optin)}B, dyn_smem_used={int(shared_bytes)}B")
    except Exception as e:
        print(f"[Device {label}] (smem caps query skipped) {e}")

    # Build metrics dict for CSV/logging
    metrics = {
        "label": label.strip(),
        "ptx": str(ptx_path.name),
        "reqntid": parse_reqntid(ptx_path),
        "regs": int(regs) if regs is not None else None,
        "shared_static": int(sh_static) if sh_static is not None else None,
        "local_bytes": int(local_bytes) if local_bytes is not None else None,
        "max_threads_per_block": int(max_threads) if max_threads is not None else None,
        "active_blocks_sm": int(active_blocks) if isinstance(active_blocks, (int, np.integer)) else (int(active_blocks) if (active_blocks is not None) else None),
        "smem_optin": int(smem_per_blk_optin) if isinstance(smem_per_blk_optin, (int, np.integer)) else (int(smem_per_blk_optin) if (smem_per_blk_optin is not None) else None),
        "dyn_smem_used": int(shared_bytes),
    }

    stream = 0
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    cuda.cuLaunchKernel(
        fn,
        int(grid_x), 1, 1,
        int(block_x), 1, 1,
        int(shared_bytes),
        int(stream),
        int(kernelParams_ptr),
        0,
    )
    end.record()
    torch.cuda.synchronize()
    kernel_ms = start.elapsed_time(end)

    # 9) Read back results
    outs_cpu = d_outs[:batch_size].cpu().numpy()

    # Static-band cell count for GCUPS
    cells_static = 0
    for i in range(batch_size):
        m = int(d_m[i].item())
        n = int(d_n[i].item())
        cells_static += diag_band_cells(m, n, CONFIG['band_width'])

    return outs_cpu, kernel_ms, cells_static, metrics


# ==================================================
# Baseline (existing Triton pipeline) for correctness cross-check
# ==================================================

def run_baseline_triton(q_list, r_list):
    task = AlignmentTask(
        align_problem='Extension',
        configuration=SolutionConfiguration(
            scoring=create_affine_scoring(CONFIG['match'], CONFIG['mismatch'], CONFIG['gap_open'], CONFIG['gap_extend']),
            pruning=create_banded_z_drop_pruning(CONFIG['band_width'], CONFIG['z_drop'])
        )
    )

    # Warm-up to JIT compile
    _ = run_alignment_task(q_list, r_list, task)
    torch.cuda.synchronize()

    start = time.perf_counter()
    results, kernel_ms, setup_ms, pack_ms, d2h_ms = run_alignment_task(q_list, r_list, task)
    end = time.perf_counter()
    wall_ms = (end - start) * 1000

    return results, kernel_ms, wall_ms


# ==================================================
# Main
# ==================================================

def main():
    print("--- PTX vs Triton (tiny dataset) ---")

    dataset_path = project_root / "datasets"
    q_all = read_fasta_as_bytes(dataset_path / "query_x10.fa")
    r_all = read_fasta_as_bytes(dataset_path / "ref_x10.fa")

    if len(q_all) == 0 or len(r_all) == 0:
        print(f"Error: dataset not found or empty in {dataset_path}")
        return

    sweep_sizes = [10, 100, 1000, 10000, 16384, 32768, 65536]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = project_root / "benchmarking" / "results"
    results_dir.mkdir(exist_ok=True)
    csv_path = results_dir / f"ptx_bench_sweep_{ts}.csv"

    # Prepare CSV
    with open(csv_path, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow([
            "n_pairs", "label", "kernel_ms", "GCUPS",
            "active_blocks_sm", "dyn_smem_used_B", "regs", "shared_static_B",
            "local_bytes", "max_threads_per_block", "reqntid", "ptx"
        ])

        for npairs in sweep_sizes:
            print(f"Running on {npairs} pairs (preview only).")
            q_list = q_all[:npairs]
            r_list = r_all[:npairs]

            # --- Baseline Triton ---
            base_res, base_k_ms, base_wall_ms = run_baseline_triton(q_list, r_list)
            print(f"[Baseline Triton] kernel_ms={base_k_ms:.3f} wall_ms={base_wall_ms:.3f}")

            # --- PTX (shared-memory) ---
            ptx_outs, ptx_k_ms, cells_static, m_shared = run_ptx_alignment(q_list, r_list)
            print(f"[PTX Shared ]    kernel_ms={ptx_k_ms:.3f} (static-band cells={cells_static/1e9:.3f} G)")
            print(f"[PTX Shared ]    GCUPS={gcups_from_cells(cells_static, ptx_k_ms):.3f}")

            # --- PTX from Triton dump (native/unconstrained) ---
            tri_native_outs, tri_native_ms, tri_native_cells, m_tri_nat = run_ptx_alignment(
                q_list, r_list,
                ptx_path=TRITON_PTX_PATH,
                use_shared=False,
                label="[PTX Triton nat]",
                carveout=-1,
                force_equal_smem=False
            )
            print(f"[PTX Triton nat] kernel_ms={tri_native_ms:.3f} (static-band cells={tri_native_cells/1e9:.3f} G)")
            print(f"[PTX Triton nat] GCUPS={gcups_from_cells(tri_native_cells, tri_native_ms):.3f}")

            # --- PTX from Triton dump (equal-occupancy: pre-reserve same dyn smem as shared) ---
            tri_eq_outs, tri_eq_ms, tri_eq_cells, m_tri_eq = run_ptx_alignment(
                q_list, r_list,
                ptx_path=TRITON_PTX_PATH,
                use_shared=False,
                label="[PTX Triton eq]",
                carveout=-1,
                force_equal_smem=True
            )
            print(f"[PTX Triton eq ] kernel_ms={tri_eq_ms:.3f} (static-band cells={tri_eq_cells/1e9:.3f} G)")
            print(f"[PTX Triton eq ] GCUPS={gcups_from_cells(tri_eq_cells, tri_eq_ms):.3f}")

            # --- Correctness check vs baseline ---
            base_arr = torch.tensor([[s, ij[0], ij[1]] for (s, ij) in base_res], dtype=torch.int32).numpy()

            def check_and_maybe_dump(ptx_outs_arr, tag):
                ok = (base_arr.shape == ptx_outs_arr.shape) and (base_arr == ptx_outs_arr).all()
                if ok:
                    print(f"[CORRECTNESS] {tag} vs Triton: PASS")
                else:
                    print(f"[CORRECTNESS] {tag} vs Triton: MISMATCH")
                    analyze_and_save_mismatches(
                        ptx_outs=ptx_outs_arr,
                        base_arr=base_arr,
                        q_list=q_list,
                        r_list=r_list,
                        results_dir=results_dir,
                        tag=f"np{npairs}_{tag.replace(' ', '_')}"
                    )

            check_and_maybe_dump(ptx_outs, "PTX(shared)")
            check_and_maybe_dump(tri_native_outs, "PTX(triton-native)")
            check_and_maybe_dump(tri_eq_outs, "PTX(triton-equal)")

            # --- Write CSV rows ---
            writer.writerow([npairs, m_shared["label"], f"{ptx_k_ms:.3f}", f"{gcups_from_cells(cells_static, ptx_k_ms):.3f}",
                             m_shared.get("active_blocks_sm"), m_shared.get("dyn_smem_used"), m_shared.get("regs"),
                             m_shared.get("shared_static"), m_shared.get("local_bytes"), m_shared.get("max_threads_per_block"), m_shared.get("reqntid"), m_shared.get("ptx")])
            writer.writerow([npairs, m_tri_nat["label"], f"{tri_native_ms:.3f}", f"{gcups_from_cells(tri_native_cells, tri_native_ms):.3f}",
                             m_tri_nat.get("active_blocks_sm"), m_tri_nat.get("dyn_smem_used"), m_tri_nat.get("regs"),
                             m_tri_nat.get("shared_static"), m_tri_nat.get("local_bytes"), m_tri_nat.get("max_threads_per_block"), m_tri_nat.get("reqntid"), m_tri_nat.get("ptx")])
            writer.writerow([npairs, m_tri_eq["label"], f"{tri_eq_ms:.3f}", f"{gcups_from_cells(tri_eq_cells, tri_eq_ms):.3f}",
                             m_tri_eq.get("active_blocks_sm"), m_tri_eq.get("dyn_smem_used"), m_tri_eq.get("regs"),
                             m_tri_eq.get("shared_static"), m_tri_eq.get("local_bytes"), m_tri_eq.get("max_threads_per_block"), m_tri_eq.get("reqntid"), m_tri_eq.get("ptx")])

        print(f"CSV saved to: {csv_path}")

if __name__ == "__main__":
    main()
