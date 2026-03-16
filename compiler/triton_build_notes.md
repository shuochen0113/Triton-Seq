# Triton Build Notes

When building Triton from source on a fresh server/container, make sure the required system build dependencies are installed first.

## Required packages

```bash
apt update
apt install -y cmake ninja-build zlib1g-dev libxml2-dev
```

## Recommended build command

From the Triton source directory:

```bash
TRITON_BUILD_TESTING=OFF pip install -e . --no-build-isolation
```

## Quick pre-check

Before building, verify:

```bash
cmake --version
ninja --version
dpkg -s zlib1g-dev libxml2-dev >/dev/null
```

## Reminder

This is especially important on a new container / new server / rebuilt environment, because these packages may not be present by default even if Python dependencies are already installed.

## March 16, 2026 rerun note

After rebuilding the March 16 compiler changes, rerun the saved compiler
experiment from the Triton-Seq root:

```bash
python benchmarks/scripts/experiment_triton_compiler.py \
  --compiler-label hack-v2 \
  --kernel opv9
```

Then inspect the newest `benchmarks/results/compiler_compare/*/summary.json` and
`kernel.ptx`.

The main expectation from the latest patches is:

- `ld_shared` should stay close to `14`
- `bar_sync` should stay low (`~7`)
- `st_shared` should drop from the pre-patch `52`
