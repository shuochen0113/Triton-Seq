#!/usr/bin/env python3
"""
Compare two saved compiler experiment runs.

Example:
  python benchmarks/scripts/compare_triton_compiler_runs.py \
      --run-a benchmarks/results/compiler_compare/.../summary.json \
      --run-b benchmarks/results/compiler_compare/.../summary.json
"""

from __future__ import annotations

import argparse
import difflib
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two Triton compiler experiment runs.")
    parser.add_argument("--run-a", type=Path, required=True, help="Path to run A summary.json or run directory.")
    parser.add_argument("--run-b", type=Path, required=True, help="Path to run B summary.json or run directory.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory for saved diff artifacts.",
    )
    parser.add_argument(
        "--diff-context",
        type=int,
        default=3,
        help="Context lines for the unified PTX diff.",
    )
    return parser.parse_args()


def resolve_summary(path: Path) -> Path:
    if path.is_dir():
        summary = path / "summary.json"
        if not summary.exists():
            raise FileNotFoundError(f"summary.json not found in {path}")
        return summary
    return path


def load_summary(path: Path) -> dict:
    return json.loads(path.read_text())


def load_ptx(summary: dict) -> tuple[Path | None, str]:
    ptx_path_str = summary.get("compiler_output", {}).get("asm_files", {}).get("ptx")
    if not ptx_path_str:
        return None, ""
    ptx_path = Path(ptx_path_str)
    if not ptx_path.exists():
        return ptx_path, ""
    return ptx_path, ptx_path.read_text(errors="replace")


def fmt_ms(value: float) -> str:
    return f"{value:.3f} ms"


def fmt_gcups(value: float) -> str:
    return f"{value:.3f} GCUPS"


def main() -> None:
    args = parse_args()
    run_a_path = resolve_summary(args.run_a)
    run_b_path = resolve_summary(args.run_b)
    summary_a = load_summary(run_a_path)
    summary_b = load_summary(run_b_path)

    label_a = summary_a["compiler_label"]
    label_b = summary_b["compiler_label"]
    kernel_a = summary_a["kernel"]
    kernel_b = summary_b["kernel"]
    timing_a = summary_a["timing"]["summary"]
    timing_b = summary_b["timing"]["summary"]
    ptx_stats_a = summary_a.get("compiler_output", {}).get("ptx_stats", {})
    ptx_stats_b = summary_b.get("compiler_output", {}).get("ptx_stats", {})

    kernel_speedup = timing_a["kernel_ms_mean"] / timing_b["kernel_ms_mean"]
    total_speedup = timing_a["total_ms_mean"] / timing_b["total_ms_mean"]

    lines = [
        "=" * 72,
        f"Run A: {label_a} ({kernel_a})",
        f"Run B: {label_b} ({kernel_b})",
        "",
        f"Kernel time : {label_a}={fmt_ms(timing_a['kernel_ms_mean'])} | {label_b}={fmt_ms(timing_b['kernel_ms_mean'])}",
        f"Total time  : {label_a}={fmt_ms(timing_a['total_ms_mean'])} | {label_b}={fmt_ms(timing_b['total_ms_mean'])}",
        f"GCUPS kernel: {label_a}={fmt_gcups(timing_a['gcups_kernel_only_mean'])} | {label_b}={fmt_gcups(timing_b['gcups_kernel_only_mean'])}",
        f"GCUPS total : {label_a}={fmt_gcups(timing_a['gcups_total_mean'])} | {label_b}={fmt_gcups(timing_b['gcups_total_mean'])}",
        f"Speedup     : kernel {kernel_speedup:.3f}x | total {total_speedup:.3f}x (A/B)",
        "",
        "PTX opcode counts:",
    ]

    stat_keys = sorted(set(ptx_stats_a) | set(ptx_stats_b))
    for key in stat_keys:
        lines.append(f"  {key}: {label_a}={ptx_stats_a.get(key, 0)} | {label_b}={ptx_stats_b.get(key, 0)}")

    ptx_path_a, ptx_a = load_ptx(summary_a)
    ptx_path_b, ptx_b = load_ptx(summary_b)
    diff_text = ""
    if ptx_a and ptx_b:
        diff_text = "\n".join(
            difflib.unified_diff(
                ptx_a.splitlines(),
                ptx_b.splitlines(),
                fromfile=str(ptx_path_a),
                tofile=str(ptx_path_b),
                n=args.diff_context,
                lineterm="",
            )
        )
        lines.extend(
            [
                "",
                f"PTX path A  : {ptx_path_a}",
                f"PTX path B  : {ptx_path_b}",
                f"PTX diff    : {'present' if diff_text else 'no textual diff'}",
            ]
        )
    else:
        lines.extend(["", "PTX diff    : unavailable (one or both PTX files missing)"])

    lines.append("=" * 72)
    report = "\n".join(lines)
    print(report)

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "comparison_report.txt").write_text(report)
        if diff_text:
            (args.output_dir / "kernel.ptx.diff").write_text(diff_text)


if __name__ == "__main__":
    main()
