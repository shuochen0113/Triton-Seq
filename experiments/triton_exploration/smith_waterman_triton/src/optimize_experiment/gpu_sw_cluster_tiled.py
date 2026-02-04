import torch
import triton
import triton.language as tl

@triton.jit
def sw_kernel_cluster(
    seq1_ptr, seq2_ptr, dp_ptr,
    m, n, match, mismatch, gap,
    seq1_stride, seq2_stride, dp_stride,
    TILE_M: tl.constexpr, TILE_N: tl.constexpr
):
    """
    Cooperative, cluster-tiled Smith-Waterman kernel.
    - The DP matrix (of size (m+1)x(n+1)) is partitioned into tiles of size TILE_M x TILE_N.
    - The grid is launched with shape (num_tiles_row, num_tiles_col), where:
          num_tiles_row = ceil(m/TILE_M)
          num_tiles_col = ceil(n/TILE_N)
    - Each block computes one tile when its wave index (tile_i + tile_j) is reached.
    - A grid-level barrier (tl.grid_barrier()) synchronizes tiles between wavefronts,
      ensuring that dependency data (the tile borders) is visible.
    """
    # Get tile (block) indices from grid
    tile_i = tl.program_id(0)  # tile row index
    tile_j = tl.program_id(1)  # tile column index

    # Compute the number of tiles in each dimension.
    num_tiles_row = tl.cdiv(m, TILE_M)
    num_tiles_col = tl.cdiv(n, TILE_N)
    num_waves = num_tiles_row + num_tiles_col - 1

    # Compute the wave index for this tile.
    wave_idx = tile_i + tile_j

    # Compute the global starting indices for this tile in the DP matrix.
    row_start = tile_i * TILE_M
    col_start = tile_j * TILE_N

    # We use a local (in-register) tile buffer to compute DP for this submatrix.
    # The tile is allocated with an extra row and column for the boundary.
    tile = tl.zeros((TILE_M + 1, TILE_N + 1), dtype=tl.int32)

    # ---- Initialize tile boundaries ----
    # Top boundary: if this tile is not in the first row, load from global dp.
    for j in range(0, TILE_N + 1):
        global_j = col_start + j
        if tile_i == 0:
            tile[0, j] = 0
        else:
            # Load from dp: dp[row_start, global_j]
            tile[0, j] = tl.load(dp_ptr + row_start * dp_stride + global_j, mask=(global_j <= n), other=0)
    # Left boundary: if this tile is not in the first column, load from global dp.
    for i in range(0, TILE_M + 1):
        global_i = row_start + i
        if tile_j == 0:
            tile[i, 0] = 0
        else:
            # Load from dp: dp[global_i, col_start]
            tile[i, 0] = tl.load(dp_ptr + global_i * dp_stride + col_start, mask=(global_i <= m), other=0)

    # ---- Wavefront scheduling: each block computes its tile when its wave is active ----
    for current_wave in range(num_waves):
        # Global barrier to synchronize all blocks between waves.
        tl.grid_barrier()  # <-- This is a grid-level (cluster) barrier. We need to call CUDA's API in Triton for this.

        if current_wave == wave_idx:
            # Now, compute the tile.
            # Loop over the tile cells (starting from 1 due to boundary).
            for i in range(1, TILE_M + 1):
                for j in range(1, TILE_N + 1):
                    global_i = row_start + i
                    global_j = col_start + j
                    # Only compute if within DP matrix bounds.
                    if (global_i <= m) and (global_j <= n):
                        # Load corresponding characters from sequences.
                        s1 = tl.load(seq1_ptr + (global_i - 1) * seq1_stride, mask=True)
                        s2 = tl.load(seq2_ptr + (global_j - 1) * seq2_stride, mask=True)
                        # Compute match/mismatch score.
                        score = tl.where(s1 == s2, match, mismatch)
                        # Compute candidates from diagonal, top, and left.
                        diag = tile[i - 1, j - 1] + score
                        up   = tile[i - 1, j] + gap
                        left = tile[i, j - 1] + gap
                        tile[i, j] = tl.maximum(tl.maximum(diag, up), tl.maximum(left, 0))
            # Write back the computed tile interior to the global DP matrix.
            for i in range(1, TILE_M + 1):
                for j in range(1, TILE_N + 1):
                    global_i = row_start + i
                    global_j = col_start + j
                    if (global_i <= m) and (global_j <= n):
                        tl.store(dp_ptr + global_i * dp_stride + global_j, tile[i, j])
        # End of current wave. All blocks synchronize before the next wave.
    # End of kernel.