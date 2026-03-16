#!/usr/bin/env python3
"""
Benchmark a Triton-Seq kernel with the currently installed Triton compiler.

This script is designed for the workflow where the user manually switches
between compiler installs on a GPU server, e.g.:

  python -m pip install -e /path/to/upstream-triton
  python benchmarks/scripts/experiment_triton_compiler.py \
      --compiler-label upstream --kernel opv6

  python -m pip install -e /path/to/triton-sw-hack
  python benchmarks/scripts/experiment_triton_compiler.py \
      --compiler-label hack-v2 --kernel opv6

With the custom compiler installed, OPv9 can also be benchmarked:

  python benchmarks/scripts/experiment_triton_compiler.py \
      --compiler-label hack-v2 --kernel opv9
"""

from __future__ import annotations

import argparse
import inspect
import json
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import torch
import triton
import triton.language as tl

from kernel.sw_kernel import sw_kernel
from utils.io import read_fasta_as_bytes
from utils.packing import pack_batch_into_buffers

try:
    from kernel.experimental.local_dp_kernel_OPv9_smem import sw_kernel_smem
except Exception:
    sw_kernel_smem = None


NEG_INF = -10_000_000
MATCH = 1
MISMATCH = -4
GAP_OPEN = -6
GAP_EXTEND = -2
Z_DROP = 400
MAX_SEQ_LEN = 10_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark one Triton-Seq kernel with the currently installed Triton compiler."
    )
    parser.add_argument("--compiler-label", required=True, help="Short label for this installed compiler.")
    parser.add_argument(
        "--compiler-root",
        type=Path,
        default=None,
        help="Optional path to the Triton compiler checkout used for this run.",
    )
    parser.add_argument(
        "--kernel",
        choices=("opv6", "opv9"),
        default="opv6",
        help="Kernel to benchmark. Upstream Triton only supports opv6.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=project_root / "datasets" / "standard",
        help="Directory containing query.fa and ref.fa.",
    )
    parser.add_argument("--query-file", type=Path, default=None, help="Override query FASTA path.")
    parser.add_argument("--ref-file", type=Path, default=None, help="Override reference FASTA path.")
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Limit the dataset to the first N pairs.",
    )
    parser.add_argument("--batch-size", type=int, default=16384, help="Pairs per launch.")
    parser.add_argument("--repeats", type=int, default=5, help="Timed repetitions.")
    parser.add_argument("--band", type=int, default=751, help="Logical band width.")
    parser.add_argument("--block", type=int, default=256, help="Threads per block.")
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Physical stride. Defaults to ceil(band / 32) * 32.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=project_root / "benchmarks" / "results" / "compiler_compare",
        help="Base directory for all experiment outputs.",
    )
    return parser.parse_args()


def run_cmd(cmd: list[str], cwd: Path | None = None) -> str | None:
    try:
        out = subprocess.check_output(cmd, cwd=str(cwd) if cwd else None, stderr=subprocess.DEVNULL)
    except Exception:
        return None
    return out.decode("utf-8", errors="replace").strip()


def resolve_dataset_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.query_file and args.ref_file:
        return args.query_file.resolve(), args.ref_file.resolve()

    query_path = args.dataset_dir / "query.fa"
    ref_path = args.dataset_dir / "ref.fa"
    if query_path.exists() and ref_path.exists():
        return query_path.resolve(), ref_path.resolve()

    fallback_q = project_root / "datasets" / "small" / "query_small.fa"
    fallback_r = project_root / "datasets" / "small" / "ref_small.fa"
    return fallback_q.resolve(), fallback_r.resolve()


def alloc_seq_bufs(n_pairs: int, max_seq_len: int, device: str = "cuda") -> dict[str, torch.Tensor]:
    max_raw = n_pairs * ((max_seq_len + 7) & ~7)
    return {
        "raw_u8": torch.empty(max_raw, dtype=torch.uint8, device=device),
        "packed_u32": torch.empty(max_raw // 8, dtype=torch.int32, device=device),
        "lengths": torch.empty(n_pairs, dtype=torch.int32, device=device),
        "offsets": torch.empty(n_pairs, dtype=torch.int32, device=device),
    }


def pack_and_ptrs(
    seqs: list[bytes], bufs: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    pack_batch_into_buffers(seqs, bufs, {})
    n_pairs = len(seqs)
    ptrs = bufs["packed_u32"].data_ptr() + bufs["offsets"][:n_pairs].to(torch.int64) * 4
    lens = bufs["lengths"][:n_pairs]
    return ptrs, lens


def kernel_kwargs(band: int, block: int, stride: int) -> dict[str, Any]:
    return {
        "match_score": MATCH,
        "mismatch_score": MISMATCH,
        "gap_open_penalty": GAP_OPEN,
        "gap_extend_penalty": GAP_EXTEND,
        "drop_threshold": Z_DROP,
        "SCORING_MODEL": "AFFINE",
        "PRUNING_BAND": "STATIC",
        "PRUNING_DROP": "ZDROP",
        "IS_EXTENSION": True,
        "STRIDE": stride,
        "BAND": band,
        "BLOCK": block,
    }


def build_run_dir(args: argparse.Namespace) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = f"{stamp}_{args.compiler_label}_{args.kernel}"
    run_dir = args.output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def get_repo_info() -> dict[str, Any]:
    return {
        "project_git_head": run_cmd(["git", "rev-parse", "HEAD"], cwd=project_root),
        "project_git_branch": run_cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=project_root),
        "compiler_submodule_git_head": run_cmd(
            ["git", "-C", str(project_root / "compiler" / "triton"), "rev-parse", "HEAD"]
        ),
        "compiler_submodule_git_branch": run_cmd(
            ["git", "-C", str(project_root / "compiler" / "triton"), "rev-parse", "--abbrev-ref", "HEAD"]
        ),
    }


def select_kernel(args: argparse.Namespace):
    if args.kernel == "opv6":
        return sw_kernel
    if args.kernel == "opv9":
        if sw_kernel_smem is None:
            raise RuntimeError("OPv9 kernel module could not be imported.")
        if not hasattr(tl, "allocate_shared"):
            raise RuntimeError(
                "Installed Triton does not expose tl.allocate_shared; OPv9 requires the hack v2 compiler."
            )
        return sw_kernel_smem
    raise ValueError(f"Unsupported kernel: {args.kernel}")


def kernel_source_path(kernel_obj: Any) -> str | None:
    fn = getattr(kernel_obj, "fn", None)
    if fn is None:
        return None
    try:
        return str(Path(inspect.getsourcefile(fn)).resolve())
    except Exception:
        return None


def compiled_kernel_to_files(compiled_kernel: Any, run_dir: Path) -> dict[str, Any]:
    asm = getattr(compiled_kernel, "asm", {})
    saved = {}
    for stage, payload in asm.items():
        ext = {
            "ttir": ".ttir",
            "ttgir": ".ttgir",
            "llir": ".ll",
            "ptx": ".ptx",
            "cubin": ".cubin",
            "sass": ".sass",
        }.get(stage, f".{stage}")
        out_path = run_dir / f"kernel{ext}"
        if isinstance(payload, bytes):
            out_path.write_bytes(payload)
        else:
            out_path.write_text(str(payload))
        saved[stage] = str(out_path)
    return saved


def summarize_ptx(ptx_text: str) -> dict[str, int]:
    stats = {
        "ld_shared": len(re.findall(r"\bld\.shared\b", ptx_text)),
        "st_shared": len(re.findall(r"\bst\.shared\b", ptx_text)),
        "ld_global": len(re.findall(r"\bld\.global\b", ptx_text)),
        "st_global": len(re.findall(r"\bst\.global\b", ptx_text)),
        "shared_decl": len(re.findall(r"\.shared\b", ptx_text)),
        "bar_sync": len(re.findall(r"\bbar\.sync\b", ptx_text)),
    }
    return stats


def metadata_to_dict(metadata: Any) -> dict[str, Any]:
    result = {}
    for key in dir(metadata):
        if key.startswith("_"):
            continue
        try:
            value = getattr(metadata, key)
        except Exception:
            continue
        if callable(value):
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            result[key] = value
        elif isinstance(value, (tuple, list)):
            result[key] = list(value)
        else:
            result[key] = str(value)
    return result


def detect_max_seq_len(q_seqs: list[bytes], r_seqs: list[bytes]) -> int:
    if not q_seqs or not r_seqs:
        return MAX_SEQ_LEN
    return max(
        8,
        max(len(seq) for seq in q_seqs),
        max(len(seq) for seq in r_seqs),
    )


def time_kernel(
    kernel_name: str,
    kernel_obj: Any,
    q_seqs: list[bytes],
    r_seqs: list[bytes],
    args: argparse.Namespace,
    stride: int,
) -> dict[str, Any]:
    batch_size = min(args.batch_size, len(q_seqs))
    max_seq_len = detect_max_seq_len(q_seqs, r_seqs)
    q_bufs = alloc_seq_bufs(batch_size, max_seq_len)
    r_bufs = alloc_seq_bufs(batch_size, max_seq_len)
    kwargs = kernel_kwargs(args.band, args.block, stride)

    total_cells = sum(len(q) * len(r) for q, r in zip(q_seqs, r_seqs))
    n_batches = (len(q_seqs) + batch_size - 1) // batch_size

    neg_inf = NEG_INF
    outs = torch.zeros((batch_size, 3), dtype=torch.int32, device="cuda")
    Hbuf = Ebuf = Fbuf = None
    if kernel_name == "opv6":
        Hbuf = torch.full((batch_size, 3 * stride), neg_inf, dtype=torch.int32, device="cuda")
        Ebuf = torch.full((batch_size, 2 * stride), neg_inf, dtype=torch.int32, device="cuda")
        Fbuf = torch.full((batch_size, 2 * stride), neg_inf, dtype=torch.int32, device="cuda")

    q_ptrs, r_ptrs, m_arr, n_arr = None, None, None, None
    first_q = q_seqs[:batch_size]
    first_r = r_seqs[:batch_size]
    q_ptrs, m_arr = pack_and_ptrs(first_q, q_bufs)
    r_ptrs, n_arr = pack_and_ptrs(first_r, r_bufs)

    if kernel_name == "opv6":
        compiled_kernel = kernel_obj.warmup(
            q_ptrs,
            r_ptrs,
            m_arr,
            n_arr,
            outs[: len(first_q)],
            Hbuf[: len(first_q)],
            Fbuf[: len(first_q)],
            Ebuf[: len(first_q)],
            grid=(len(first_q),),
            **kwargs,
        )
    else:
        compiled_kernel = kernel_obj.warmup(
            q_ptrs,
            r_ptrs,
            m_arr,
            n_arr,
            outs[: len(first_q)],
            grid=(len(first_q),),
            **kwargs,
        )
    torch.cuda.synchronize()

    repeat_records = []
    for repeat_idx in range(args.repeats):
        kernel_ms_total = 0.0
        init_ms_total = 0.0

        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(q_seqs))
            q_batch = q_seqs[start:end]
            r_batch = r_seqs[start:end]
            n_pairs = end - start

            q_ptrs, m_arr = pack_and_ptrs(q_batch, q_bufs)
            r_ptrs, n_arr = pack_and_ptrs(r_batch, r_bufs)
            torch.cuda.synchronize()

            if kernel_name == "opv6":
                init_start = torch.cuda.Event(enable_timing=True)
                init_end = torch.cuda.Event(enable_timing=True)
                kernel_start = torch.cuda.Event(enable_timing=True)
                kernel_end = torch.cuda.Event(enable_timing=True)

                init_start.record()
                Hbuf[:n_pairs].fill_(neg_inf)
                Ebuf[:n_pairs].fill_(neg_inf)
                Fbuf[:n_pairs].fill_(neg_inf)
                init_end.record()

                kernel_start.record()
                kernel_obj[n_pairs,](
                    q_ptrs,
                    r_ptrs,
                    m_arr,
                    n_arr,
                    outs[:n_pairs],
                    Hbuf[:n_pairs],
                    Fbuf[:n_pairs],
                    Ebuf[:n_pairs],
                    **kwargs,
                )
                kernel_end.record()
                torch.cuda.synchronize()

                init_ms_total += init_start.elapsed_time(init_end)
                kernel_ms_total += kernel_start.elapsed_time(kernel_end)
            else:
                kernel_start = torch.cuda.Event(enable_timing=True)
                kernel_end = torch.cuda.Event(enable_timing=True)

                kernel_start.record()
                kernel_obj[n_pairs,](
                    q_ptrs,
                    r_ptrs,
                    m_arr,
                    n_arr,
                    outs[:n_pairs],
                    **kwargs,
                )
                kernel_end.record()
                torch.cuda.synchronize()

                kernel_ms_total += kernel_start.elapsed_time(kernel_end)

        repeat_records.append(
            {
                "repeat": repeat_idx,
                "kernel_ms": kernel_ms_total,
                "init_ms": init_ms_total,
                "total_ms": kernel_ms_total + init_ms_total,
                "gcups_kernel_only": (total_cells / (kernel_ms_total / 1000.0)) / 1e9,
                "gcups_total": (total_cells / ((kernel_ms_total + init_ms_total) / 1000.0)) / 1e9,
            }
        )

    kernel_values = [record["kernel_ms"] for record in repeat_records]
    init_values = [record["init_ms"] for record in repeat_records]
    total_values = [record["total_ms"] for record in repeat_records]

    return {
        "compiled_kernel": compiled_kernel,
        "repeat_records": repeat_records,
        "summary": {
            "n_pairs": len(q_seqs),
            "n_batches": n_batches,
            "batch_size": batch_size,
            "total_cells": total_cells,
            "kernel_ms_mean": mean(kernel_values),
            "kernel_ms_std": pstdev(kernel_values) if len(kernel_values) > 1 else 0.0,
            "init_ms_mean": mean(init_values),
            "init_ms_std": pstdev(init_values) if len(init_values) > 1 else 0.0,
            "total_ms_mean": mean(total_values),
            "total_ms_std": pstdev(total_values) if len(total_values) > 1 else 0.0,
            "gcups_kernel_only_mean": mean(record["gcups_kernel_only"] for record in repeat_records),
            "gcups_total_mean": mean(record["gcups_total"] for record in repeat_records),
        },
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Run this script on the GPU server.")

    stride = args.stride if args.stride is not None else ((args.band + 31) // 32) * 32
    kernel_obj = select_kernel(args)
    run_dir = build_run_dir(args)

    query_path, ref_path = resolve_dataset_paths(args)
    q_seqs = read_fasta_as_bytes(query_path)
    r_seqs = read_fasta_as_bytes(ref_path)
    if len(q_seqs) != len(r_seqs):
        raise RuntimeError(f"Dataset size mismatch: {len(q_seqs)} queries vs {len(r_seqs)} refs.")
    if args.max_pairs is not None:
        q_seqs = q_seqs[: args.max_pairs]
        r_seqs = r_seqs[: args.max_pairs]
    if not q_seqs:
        raise RuntimeError("No sequence pairs found.")

    started_at = datetime.now(timezone.utc)
    timed = time_kernel(args.kernel, kernel_obj, q_seqs, r_seqs, args, stride)
    finished_at = datetime.now(timezone.utc)

    asm_files = compiled_kernel_to_files(timed["compiled_kernel"], run_dir)
    ptx_text = ""
    ptx_path = asm_files.get("ptx")
    if ptx_path is not None:
        ptx_text = Path(ptx_path).read_text(errors="replace")

    device_index = torch.cuda.current_device()
    device_props = torch.cuda.get_device_properties(device_index)
    summary = {
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "compiler_label": args.compiler_label,
        "compiler_root": str(args.compiler_root.resolve()) if args.compiler_root else None,
        "kernel": args.kernel,
        "kernel_source": kernel_source_path(kernel_obj),
        "dataset": {
            "query_file": str(query_path),
            "ref_file": str(ref_path),
            "max_pairs": args.max_pairs,
        },
        "config": {
            "band": args.band,
            "block": args.block,
            "stride": stride,
            "repeats": args.repeats,
            "batch_size": args.batch_size,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "triton_version": getattr(triton, "__version__", None),
            "triton_file": str(Path(triton.__file__).resolve()),
            "tl_allocate_shared_available": hasattr(tl, "allocate_shared"),
            "cuda_device_index": device_index,
            "cuda_device_name": device_props.name,
            "cuda_capability": [device_props.major, device_props.minor],
            "total_memory_bytes": device_props.total_memory,
        },
        "git": get_repo_info(),
        "compiler_output": {
            "metadata": metadata_to_dict(timed["compiled_kernel"].metadata),
            "asm_files": asm_files,
            "ptx_stats": summarize_ptx(ptx_text) if ptx_text else {},
        },
        "timing": {
            "repeats": timed["repeat_records"],
            "summary": timed["summary"],
        },
    }

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print("=" * 72)
    print(f"Run directory : {run_dir}")
    print(f"Compiler label: {args.compiler_label}")
    print(f"Kernel        : {args.kernel}")
    print(f"Dataset pairs : {timed['summary']['n_pairs']:,}")
    print(f"Kernel ms     : {timed['summary']['kernel_ms_mean']:.3f} ± {timed['summary']['kernel_ms_std']:.3f}")
    print(f"Init ms       : {timed['summary']['init_ms_mean']:.3f} ± {timed['summary']['init_ms_std']:.3f}")
    print(f"Total ms      : {timed['summary']['total_ms_mean']:.3f} ± {timed['summary']['total_ms_std']:.3f}")
    print(f"GCUPS kernel  : {timed['summary']['gcups_kernel_only_mean']:.3f}")
    print(f"GCUPS total   : {timed['summary']['gcups_total_mean']:.3f}")
    if ptx_text:
        print(f"PTX stats     : {summary['compiler_output']['ptx_stats']}")
    print("=" * 72)


if __name__ == "__main__":
    main()
