# Summary of Codon/Seq & Triton EXTZ Exploration

## 1. Review of Codon Language & Toolchain
- Studied the official Codon tutorial and advanced GPU guide  
  (https://docs.exaloop.io/codon/, “Advanced GPU Programming”).  
- Noted that Codon is a Python-inspired AOT compiler with optional OpenMP-style CPU parallelism (`@par`) and minimal CUDA backend (`@gpu.kernel`).

## 2. CPU Reference Implementation (`extz_cpu.codon`)
- Ported the AGAThA-style Smith–Waterman + extension + Z-drop algorithm to Codon:  
  - **Affine-gap** (gap open + extend)  
  - **Banded anti-diagonal sweep** with dynamic band center  
  - **Z-drop early termination**  
- Exposed parameters: `match, mismatch, gapo, gape, Z, band`  
- Used OpenMP-style parallelism (`@par(schedule="dynamic", chunk_size=1, num_threads)`) to align multiple (query, ref) pairs concurrently.  
- Verified bit-exact results on 15 test pairs against Triton implementation.

## 3. Prototype of Codon GPU Version
- Attempted a `@gpu.kernel` version in Codon mirroring Triton’s logic and buffer rotations.  
- Discovered key limitations:
  - **No** dedicated `codon build -gpu` CLI flag (use `codon run` or `codon build` only).  
  - GPU API in Codon uses `@gpu.kernel`, but there is no high-level host/device memory allocation tutorials.
  - Codon’s only GPU examples in docs are elementary (vector add, Mandelbrot), so whether Codon can write GPU sequence alignment kernels remains questionable.

## 4. Seq-plugin Sequence Alignment Exploration
- **Plugin installation & usage**  
  - Installed `seq` via `curl`/untar into `/codon/build/codon/lib/codon/plugins/seq`.  
  - Default FASTA reader requires `.fai` index; bypassed with `FASTA(path, fai=False)`.  
  - Invoked with `codon run -plugin seq seq_test.codon`.

- **Built-in alignment API**  
  1. `s1 @ s2`  
     - Global (Needleman–Wunsch) alignment with fixed match/mismatch + single affine-gap model.  
  2. `s1.align(s2, a, b, gapo, gape, gapo2, gape2)`  
     - Two-piece affine gap:  
       - **Primary**: `gapo + k·gape`  
       - **Secondary**: `gapo2 + k·gape2`  
     - Chooses the lower cost at each gap length.

- **Limitations & conclusions**  
  - Supports **only** global alignment with two-phase affine gaps.  
  - Cannot switch to Smith–Waterman (local), semi-global, banded or Z-drop variants via API.  
  - To implement SW or extZ in Seq, one must write a custom DP routine by hand. (in codon?)

## 5. Benchmarking (AGAThA's dataset, 16384 pairs)
- **Triton**: kernel time 428.479 ms, total time 26690.94 ms
- **ksw2_extz**: 34354.06 ms
- **seq**: 167080.89 ms
- **codon** (cpu, 8 threads parallel): 3024048.61 ms
