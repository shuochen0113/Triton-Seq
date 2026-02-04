# prototype/utils/packing_kernel_OPv6.py
"""
2025-07-24
GPU 4-bit DNA packing - v4
"""
from typing import List, Tuple
import torch
import triton
import triton.language as tl
import numpy as np

# ---------- ASCII -> 4-bit LUT (unchanged) ----------
_DNA_LUT_CPU = torch.zeros(256, dtype=torch.uint8)
for c, v in {**{k: 1 for k in "Aa"},
             **{k: 2 for k in "Cc"},
             **{k: 3 for k in "Gg"},
             **{k: 4 for k in "Tt"},
             **{k: 14 for k in "Nn"}}.items():
    _DNA_LUT_CPU[ord(c)] = v
_DNA_LUT_GPU = _DNA_LUT_CPU.cuda(non_blocking=True)

# ---------- Triton kernel : 8xuint8 -> 1xuint32 (unchanged) ----------
@triton.jit
def _pack_kernel(src_u8, lut, dst_i32, total_words: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """The core Triton kernel that packs 8 bytes into a single 32-bit integer."""
    pid = tl.program_id(0)
    block_start_wid = pid * BLOCK_SIZE
    w_offsets = block_start_wid + tl.arange(0, BLOCK_SIZE)
    w_mask = w_offsets < total_words
    c_offsets = w_offsets[:, None] * 8 + tl.arange(0, 8)[None, :]
    chars = tl.load(src_u8 + c_offsets, mask=w_mask[:, None], other=0)
    nibb = tl.load(lut + chars, mask=w_mask[:, None], other=0)
    shift_amouts = (tl.arange(0, 8)[None, :] * 4).to(tl.uint32)
    word_u32 = tl.sum(nibb.to(tl.uint32) << shift_amouts, axis=1)
    dst_u32_ptr = tl.cast(dst_i32, tl.pointer_type(tl.uint32))
    tl.store(dst_u32_ptr + w_offsets, word_u32, mask=w_mask)

def pack_batch_into_buffers(
    seqs_batch: List[str],
    device_buffers: dict,
    timing_events: dict 
):
    """
    Packs a batch of sequences directly into pre-allocated GPU buffers.
    This function is designed to be called within a CUDA stream for pipelining.

    Args:
        seqs_batch (List[str]): The batch of sequences to pack.
        device_buffers (dict): A dictionary containing pre-allocated tensors on GPU, e.g.,
                               'raw_u8', 'packed_u32', 'lengths', 'offsets'.
    """
    if not seqs_batch:
        return

    # # 1. CPU-side preparation - this part remains the same and is very fast for a batch
    # encoded_seqs = [s.encode('ascii') for s in seqs_batch]
    # lengths_b = np.array([len(s) for s in encoded_seqs], dtype=np.int32)
    # padded_lengths_b = (lengths_b + 7) & ~7
    
    # total_size = np.sum(padded_lengths_b)
    # offsets_b = np.zeros(len(seqs_batch) + 1, dtype=np.int64)
    # offsets_b[1:] = np.cumsum(padded_lengths_b)
    
    # buf = bytearray(total_size)
    # buf_view = memoryview(buf)
    # for i, enc_seq in enumerate(encoded_seqs):
    #     start = offsets_b[i]
    #     buf_view[start:start + len(enc_seq)] = enc_seq
    
    # # We create a temporary CPU tensor to facilitate the H2D copy
    # # cpu_tensor = torch.from_numpy(np.frombuffer(buf, dtype=np.uint8))
    # # ======= 2025-07-24: use pinned memory for faster H2D transfer =======
    # cpu_tensor = torch.from_numpy(np.frombuffer(buf, dtype=np.uint8)).pin_memory()
    # # ====================================================================

    # --- 1. OPTIMIZED CPU-side preparation ---

    # Optimization 1: Use np.fromiter with map(len) to replace the slow list comprehension.
    # This is significantly faster as the looping happens at the C level.
    lengths_b = np.fromiter(map(len, seqs_batch), dtype=np.int32, count=len(seqs_batch))
    
    # The .encode() list comprehension is now completely gone, as we receive bytes.
    # encoded_seqs = [s.encode('ascii') for s in seqs_batch] # REMOVED

    padded_lengths_b = (lengths_b + 7) & ~7
    total_size = np.sum(padded_lengths_b)
    offsets_b = np.zeros(len(seqs_batch) + 1, dtype=np.int64)
    offsets_b[1:] = np.cumsum(padded_lengths_b)
    
    # The bytearray filling loop is the remaining bottleneck in pure Python.
    # While it's hard to eliminate completely without C extensions or multiprocessing,
    # the optimizations above have already removed the most expensive parts.
    buf = bytearray(total_size)
    buf_view = memoryview(buf)
    for i, seq in enumerate(seqs_batch):
        start = offsets_b[i]
        buf_view[start:start + len(seq)] = seq
    
    # This part remains the same, using pinned memory
    cpu_tensor = torch.from_numpy(np.frombuffer(buf, dtype=np.uint8)).pin_memory()

    # --- Timing for H2D Transfer ---
    h2d_start = torch.cuda.Event(enable_timing=True)
    h2d_end = torch.cuda.Event(enable_timing=True)
    h2d_start.record()
    
    # 2. Asynchronous H2D copy into the pre-allocated buffer
    # We only copy the part of the buffer we need for this batch
    raw_u8_gpu = device_buffers['raw_u8'][:total_size]
    raw_u8_gpu.copy_(cpu_tensor, non_blocking=True)

    # This event is recorded on the same stream, so it will wait for the copy to finish
    h2d_end.record() 
    timing_events['h2d'] = (h2d_start, h2d_end)

    # --- Timing for Pack Kernel ---
    pack_start = torch.cuda.Event(enable_timing=True)
    pack_end = torch.cuda.Event(enable_timing=True)
    pack_start.record()

    # 3. Launch Triton kernel using the pre-allocated buffers
    total_words = int(total_size // 8)
    packed_gpu = device_buffers['packed_u32'][:total_words]
    
    grid_size = triton.cdiv(total_words, 256)
    if grid_size > 0:
        _pack_kernel[grid_size,](
            src_u8=raw_u8_gpu, 
            lut=_DNA_LUT_GPU, 
            dst_i32=packed_gpu,
            total_words=total_words, 
            BLOCK_SIZE=256
        )

    pack_end.record()
    timing_events['pack_kernel'] = (pack_start, pack_end)

    # 4. Prepare metadata tensors on GPU
    batch_size = len(seqs_batch)
    lengths_gpu = device_buffers['lengths'][:batch_size]
    # ====== 2025-07-24: use pinned memory for faster D2H transfer =======
    lengths_gpu.copy_(torch.from_numpy(lengths_b).pin_memory(), non_blocking=True)

    offsets_w_gpu = device_buffers['offsets'][:batch_size]
    offsets_w_gpu.copy_(torch.from_numpy(offsets_b[:-1] // 8).to(dtype=torch.int32).pin_memory(), non_blocking=True)
    # ==================================================================
    
    # The function no longer returns anything, as it modifies buffers in-place.