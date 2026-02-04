# 📄 Seq: A High-Performance Language for Bioinformatics

## 📌 Basic Information
- **Title**: Seq: A High-Performance Language for Bioinformatics  
- **Authors**: Ariya Shajii, Ibrahim Numanagić, Riyadh Baghdadi, Bonnie Berger, Saman Amarasinghe  
- **Published at**: OOPSLA 2019  
- **Link**: https://doi.org/10.1145/3360551  

---

## Motivation
- **Problem**: Biological sequencing data is growing exponentially, but computing power (Moore's Law) is slowing down.
- **Pain Point**: Current bioinformatics tools are either:
  - High-level but slow (Python, R), or
  - Fast but hard to write/maintain (C/C++)
- **Goal**: Make writing high-performance bioinformatics software as easy as writing Python.

---

## Introduction to Seq
- **Type**: Domain-Specific Language (DSL)
- **Syntax**: Subset of Python
- **Compiler Backend**: LLVM
- **Runtime**: Minimal (~200 LOC), uses Boehm GC
- **Key Claim**: Combines Python's productivity with C/C++'s performance
- **Performance**:
  - Up to 160× faster than Python
  - Up to 2× faster than hand-optimized C++
  - With parallelism: up to 650× faster than Python

---

## Section 2: Primer on Computational Genomics
- **Core Data Types**:
  - DNA sequences: strings over Σ = {A, C, G, T}
  - Reads: ~100bp substrings from sequencing machines
  - Reference genome: ~3Gbp consensus sequence
  - k-mers: contiguous substrings of fixed length k

- **Key Tasks**:
  - **Sequence alignment**: Finding minimal edit distance match in reference genome (uses k-mer seeding + Smith-Waterman)
  - **De novo assembly**: Uses de Bruijn graphs from k-mers to reconstruct genome
  - **Common Ops**: k-merization, reverse complement, indexing, pattern matching

---

## Section 3: k-merization & Seeding Example

**Seq Code Example**:
```python
from genomeindex import *
type K = Kmer[20]

def process(kmer: K, index: GenomeIndex[K]):
    prefetch index[kmer], index[~kmer]
    hits_fwd = index[kmer]
    hits_rev = index[~kmer]

index = GenomeIndex[K]("reference.fa")
stride = 10
(fastq("reads.fq") |> kmers[K](stride) |> process(index))
```

**Compared to C++**:
- No manual memory or file I/O management
- No custom reverse complement logic
- Pipeline fusion and prefetching automatically applied

**Key Takeaway**:
> Same bioinformatics logic, 10× less code, and runs faster than optimized C++.

---

## Section 4: Language Design & Implementation

### 4.1 Statically-Typed Python
- Full static typing system
- Type inference via bi-directional Hindley-Milner algorithm
- Duck typing at compile-time
- Generic types and functions (`def f[T](x: T) -> T`)
- Compiled to LLVM IR (e.g. `Kmer[20]` → `i40`)

### 4.2 Coroutines and Generators
- Generators implemented as LLVM coroutines
- Automatic coroutine inlining/unrolling
- `for i in range(3)` compiles to same IR as C++ loop

### 4.3 Genomics-Specific Language Features
- `seq` type: biological sequence object
- `Kmer[n]` type: fixed-length 2-bit-encoded sequences
- `~x`: reverse complement operator
- `|>`: pipeline operator
- `||>`: parallel pipeline operator
- `match`: supports sequence pattern matching (e.g. spaced seeds)
- `cdef`: C interop
- `extend`: compile-time type extensions (e.g. `extend int:`)

---

## Section 5: Optimizations

### 5.1 2-bit k-mer Encoding
- Maps `Kmer[k]` → `i(2k)` LLVM type
- Reverse complement via lookup tables (4-mer based)

### 5.2 Parallelism with `||>`
- Pipeline stages compiled to OpenMP tasks via Tapir
- Nested and streamed parallelism

### 5.3 Software Prefetching
- `__prefetch__()` method hints to compiler
- Compiler transforms function into coroutine that yields
- Overlaps memory stalls with other coroutines
- Enables dynamic scheduling of up to M concurrent pipelines

---

## Section 6: Evaluation

### Benchmarks:
1. **Computer Language Benchmarks Game**
2. **Custom Microbenchmarks (RC, CpG, 16-mer)**
3. **Real Tools (SNAP, SGA)**

### Speedup Summary:
| Seq Type | Over Python | Over C++ |
|----------|-------------|----------|
| Pythonic Seq | 11–100× | Slightly worse |
| Idiomatic Seq | **Up to 160×** | **Up to 2× faster** |
| With Parallelism | **Up to 650×** | **2×** |

### Key Insight:
> Just adding `prefetch` and `||>` yields 2× speedup over highly-tuned C++.

### Real-world Tools:
- SNAP, SGA reimplemented in Seq
- Show similar or better performance with much less code

---

## Conclusion & Key Takeaways
- Seq is a DSL combining Pythonic syntax and high-performance compilation.
- It's tailored to genomics: optimized types, reverse complement, pipelining, and indexing.
- Real-world usage shows **Seq is not only easier to write, but often faster than C++.**

> "Seq opens the door to democratizing high-performance bioinformatics computing."

---

