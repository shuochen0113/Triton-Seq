# Datasets

This directory contains test datasets for Triton-Seq benchmarking and validation.

## Available Datasets

### Small Dataset (`small/`)

**Purpose**: Quick testing and development

- **Files**: `query_small.fa`, `ref_small.fa`
- **Size**: 10 sequence pairs
- **Runtime**: ~1 second on RTX 4090
- **Use case**: Development, CI/CD, quick sanity checks

### Standard Dataset (`standard/`)

**Purpose**: Standard benchmark (AGAThA dataset)

- **Files**: `query.fa`, `ref.fa`
- **Size**: 16,384 sequence pairs
- **Source**: AGAThA paper dataset
- **Statistics**:
  - Average query length: ~900 bp
  - Average reference length: ~1200 bp
  - Total sequences: 32,768
- **Runtime**: ~120 ms on RTX 4090 (OPv6)
- **Use case**: Performance benchmarking, correctness validation

## Dataset Format

All datasets use standard FASTA format:

```
>sequence_1
ATCGATCGATCGATCG...
>sequence_2
GCTAGCTAGCTAGCTA...
...
```

## Generating Larger Datasets

For scalability testing, use the provided script:

```bash
# Generate 10x scaled dataset
python generate_scaled.py --scale 10 --output large/

# Supported scales
python generate_scaled.py --scale 5   # 81,920 pairs
python generate_scaled.py --scale 10  # 163,840 pairs
python generate_scaled.py --scale 50  # 819,200 pairs
python generate_scaled.py --scale 100 # 1,638,400 pairs
```

**Note**: Large datasets may require significant GPU memory. Adjust batch size accordingly.

## Git LFS

Large FASTA files (>50MB) use Git LFS (Large File Storage). When cloning the repository, ensure Git LFS is installed:

```bash
# Install Git LFS
git lfs install

# Pull LFS files
git lfs pull
```

## Custom Datasets

To use your own datasets:

1. Prepare FASTA files (query and reference)
2. Place in a new subdirectory (e.g., `datasets/custom/`)
3. Update benchmark scripts to point to your data

```python
from src.api import align_sequences
from src.utils.io import load_fasta

queries = load_fasta("datasets/custom/my_query.fa")
references = load_fasta("datasets/custom/my_ref.fa")
results = align_sequences(queries, references)
```

## Dataset Statistics

You can analyze dataset statistics:

```bash
python -c "
from src.utils.io import load_fasta
queries = load_fasta('datasets/standard/query.fa')
refs = load_fasta('datasets/standard/ref.fa')
print(f'Pairs: {len(queries)}')
print(f'Avg query len: {sum(len(q) for q in queries) / len(queries):.0f}')
print(f'Avg ref len: {sum(len(r) for r in refs) / len(refs):.0f}')
"
```

## Storage Requirements

| Dataset | Pairs | File Size | GPU Memory (batch=2048) |
|---------|-------|-----------|-------------------------|
| Small | 10 | ~1 KB | ~10 MB |
| Standard | 16,384 | ~42 MB | ~1.5 GB |
| 10x | 163,840 | ~420 MB | ~15 GB |
| 100x | 1,638,400 | ~4.2 GB | ~150 GB* |

*Requires multiple GPUs or reduced batch size