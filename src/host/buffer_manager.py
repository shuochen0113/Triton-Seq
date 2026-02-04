# prototype/host/buffer_manager_OPv6.py

"""
2025-08-18
Manages the creation of pre-allocated buffer sets for the asynchronous pipeline.
"""
import torch
from typing import List, Dict, Any, Tuple

def create_pipeline_resources(
    kernel_name: str, 
    cfg: Any, 
    n_streams: int, 
    batch_size: int, 
    max_seq_len: int,
    device: str = 'cuda'
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Pre-allocates all necessary host and device buffers for a multi-stream pipeline.

    Returns:
        A tuple of (host_buffers, device_buffers), where each is a list of
        dictionaries, with one dictionary per stream.
    """
    host_buffers = []
    device_buffers = []
    
    max_batch_raw_bytes = batch_size * max_seq_len
    
    for _ in range(n_streams):
        # --- Pinned host memory for asynchronous D2H results transfer ---
        h_pinned_results = torch.empty((batch_size, 3), dtype=torch.int32).pin_memory()
        host_buffers.append({'results': h_pinned_results})

        # --- Pre-allocated device memory for one stream's workload ---
        # Sequence data buffers
        d_raw_q = torch.empty(max_batch_raw_bytes, dtype=torch.uint8, device=device)
        d_packed_q = torch.empty(max_batch_raw_bytes // 8, dtype=torch.int32, device=device)
        d_len_q = torch.empty(batch_size, dtype=torch.int32, device=device)
        d_off_q = torch.empty(batch_size, dtype=torch.int32, device=device)
        
        d_raw_r = torch.empty(max_batch_raw_bytes, dtype=torch.uint8, device=device)
        d_packed_r = torch.empty(max_batch_raw_bytes // 8, dtype=torch.int32, device=device)
        d_len_r = torch.empty(batch_size, dtype=torch.int32, device=device)
        d_off_r = torch.empty(batch_size, dtype=torch.int32, device=device)

        # DP buffers, logic depends on the kernel
        d_dp_buffers = {}
        if kernel_name == 'sw_kernel':
            band_width = cfg.pruning.params.get('band_width', 1)
            # Physical stride in int32 elements (round up to 128B alignment)
            CAP = ((band_width + 31) // 32) * 32
            neg_inf32 = -10_000_000
            # H uses 3 ring slots, E/F use 2 ring slots; int32 storage
            d_dp_buffers['Hbuf'] = torch.full((batch_size, 3 * CAP), neg_inf32, dtype=torch.int32, device=device)
            d_dp_buffers['Ebuf'] = torch.full((batch_size, 2 * CAP), neg_inf32, dtype=torch.int32, device=device)
            d_dp_buffers['Fbuf'] = torch.full_like(d_dp_buffers['Ebuf'], neg_inf32)
        elif kernel_name == 'logan_kernel':
            # Add buffer allocation logic for logan_kernel here if needed
            pass
            
        # Output buffer
        d_outs = torch.empty((batch_size, 3), dtype=torch.int32, device=device)
        
        device_buffers.append({
            'q': {'raw_u8': d_raw_q, 'packed_u32': d_packed_q, 'lengths': d_len_q, 'offsets': d_off_q},
            'r': {'raw_u8': d_raw_r, 'packed_u32': d_packed_r, 'lengths': d_len_r, 'offsets': d_off_r},
            'dp': d_dp_buffers,
            'outs': d_outs
        })

    return host_buffers, device_buffers