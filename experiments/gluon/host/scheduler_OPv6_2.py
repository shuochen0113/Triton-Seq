# prototype/host/scheduler_OPv6_2.py

import torch
import time
from typing import List, Dict, Any, Callable

from utils.packing_kernel_OPv6 import pack_batch_into_buffers
from host.buffer_manager_OPv6 import create_pipeline_resources

class PipelineScheduler:
    def __init__(self, kernel_to_run: Callable, kernel_args_template: Dict[str, Any], cfg: Any, batch_size: int, max_seq_len: int, n_streams: int = 2):
        self.kernel_to_run = kernel_to_run
        self.kernel_args_template = kernel_args_template
        self.cfg = cfg
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.n_streams = n_streams

    def _submit_batch(self, batch_idx: int, stream_idx: int, q_list: List[str], r_list: List[str], n_total_pairs: int, streams: List, events: List, device_buffers: List, host_buffers: List, timing_events: Dict):
        """Helper function to submit a single batch to a specific stream."""
        current_stream = streams[stream_idx]

        # --- CPU Work (remains the same) ---
        start, end = batch_idx * self.batch_size, min((batch_idx + 1) * self.batch_size, n_total_pairs)
        q_batch, r_batch = q_list[start:end], r_list[start:end]
        current_batch_size = len(q_batch)

        # --- Enqueue GPU Work ---
        with torch.cuda.stream(current_stream):
            # C.1 Pack and H2D (remains the same)
            pack_events = {}
            pack_batch_into_buffers(q_batch, device_buffers[stream_idx]['q'], pack_events)
            timing_events['h2d'].append(pack_events['h2d'])
            timing_events['pack'].append(pack_events['pack_kernel'])

            pack_events = {}
            pack_batch_into_buffers(r_batch, device_buffers[stream_idx]['r'], pack_events)
            timing_events['h2d'].append(pack_events['h2d'])
            timing_events['pack'].append(pack_events['pack_kernel'])

            # --- MODIFIED: Remove C.2 DP Buffer Init ---
            # This step is no longer needed as H/E/F are in smem
            # REMOVED: dp_init_start = torch.cuda.Event(enable_timing=True); dp_init_end = torch.cuda.Event(enable_timing=True)
            # REMOVED: dp_init_start.record()
            # REMOVED: neg_inf32 = -10_000_000
            # REMOVED: for buf in device_buffers[stream_idx]['dp'].values():
            # REMOVED:     if isinstance(buf, torch.Tensor):
            # REMOVED:         buf.fill_(neg_inf32)
            # REMOVED: dp_init_end.record()
            # REMOVED: timing_events['dp_init'].append((dp_init_start, dp_init_end))
            # --- END MODIFICATION ---

            # C.3 Align Kernel (logic remains, args change via _prepare_kernel_args)
            align_start = torch.cuda.Event(enable_timing=True); align_end = torch.cuda.Event(enable_timing=True)
            align_start.record()

            # _prepare_kernel_args will now create args *without* H/E/F pointers
            current_kernel_args = self._prepare_kernel_args(device_buffers[stream_idx], current_batch_size)
            # Kernel launch itself is unchanged
            self.kernel_to_run[current_batch_size,](**current_kernel_args)

            align_end.record()
            timing_events['align'].append((align_start, align_end))

            # C.4 D2H Results (remains the same)
            d2h_start = torch.cuda.Event(enable_timing=True); d2h_end = torch.cuda.Event(enable_timing=True)
            d2h_start.record()
            host_buffers[stream_idx]['results'][:current_batch_size].copy_(
                device_buffers[stream_idx]['outs'][:current_batch_size], non_blocking=True
            )
            d2h_end.record()
            timing_events['d2h'].append((d2h_start, d2h_end))

            # Record event (remains the same)
            events[stream_idx].record()

    def execute(self, q_list: List[str], r_list: List[str]):
        n_total_pairs = len(q_list)
        if n_total_pairs == 0:
            return [], 0.0, 0.0, 0.0, 0.0

        # --- 1. Initialize Pipeline Resources (remains the same, but buffer_manager allocates less) ---
        streams = [torch.cuda.Stream() for _ in range(self.n_streams)]
        events = [torch.cuda.Event() for _ in range(self.n_streams)]

        kernel_name = getattr(self.kernel_to_run, "__name__", "gluon_kernel") # Use new kernel name if needed
        host_buffers, device_buffers = create_pipeline_resources(
            kernel_name, self.cfg, self.n_streams, self.batch_size, self.max_seq_len
        )

        num_batches = (n_total_pairs + self.batch_size - 1) // self.batch_size
        all_results = [None] * n_total_pairs
        # --- MODIFIED: Remove 'dp_init' from timing_events ---
        timing_events = {'h2d': [], 'pack': [], 'align': [], 'd2h': []} # Removed 'dp_init'
        # --- END MODIFICATION ---

        submitted_batches = 0
        completed_batches = 0
        stream_batch_map = [-1] * self.n_streams

        # --- 2. Phase 1: Priming Loop (remains the same) ---
        for i in range(min(self.n_streams, num_batches)):
            self._submit_batch(i, i, q_list, r_list, n_total_pairs, streams, events, device_buffers, host_buffers, timing_events)
            stream_batch_map[i] = i
            submitted_batches += 1

        # --- 3. Phase 2: Steady-State Loop (remains the same) ---
        while completed_batches < num_batches:
            # ... (loop logic remains identical) ...
            # Process results (remains identical)
            prev_batch_idx = stream_batch_map[stream_idx]
            if prev_batch_idx != -1:
                prev_start = prev_batch_idx * self.batch_size
                # ... (result processing remains identical) ...
                all_results[i] = (int(s), (int(ii), int(j)))
                result_offset += 1
            completed_batches += 1
            stream_batch_map[stream_idx] = -1

            if submitted_batches < num_batches:
                self._submit_batch(submitted_batches, stream_idx, q_list, r_list, n_total_pairs, streams, events, device_buffers, host_buffers, timing_events)
                stream_batch_map[stream_idx] = submitted_batches
                submitted_batches += 1


        # --- 4. Timing Aggregation ---
        total_kernel_time_ms = sum(s.elapsed_time(e) for s, e in timing_events['align'])
        packing_kernel_time_ms = sum(s.elapsed_time(e) for s, e in timing_events['pack'])
        total_d2h_time_ms = sum(s.elapsed_time(e) for s, e in timing_events['d2h'])
        total_h2d_time_ms = sum(s.elapsed_time(e) for s, e in timing_events['h2d'])
        # --- MODIFIED: Remove dp_init time calculation ---
        # total_dp_init_time_ms = sum(s.elapsed_time(e) for s, e in timing_events['dp_init'])
        total_dp_init_time_ms = 0.0 # Set to zero as it's removed
        # --- END MODIFICATION ---
        total_setup_ms = total_h2d_time_ms + packing_kernel_time_ms + total_dp_init_time_ms

        return all_results, total_kernel_time_ms, total_setup_ms, packing_kernel_time_ms, total_d2h_time_ms

    def _prepare_kernel_args(self, device_buffers_set, batch_size):
        # Q/R pointers, lengths, offsets (remain the same)
        # Need to ensure q_ptrs/r_ptrs calculation is correct for Gluon kernel signature if it differs
        q_ptrs = device_buffers_set['q']['packed_u32'].data_ptr() + (device_buffers_set['q']['offsets'][:batch_size].to(torch.int64) * 4)
        r_ptrs = device_buffers_set['r']['packed_u32'].data_ptr() + (device_buffers_set['r']['offsets'][:batch_size].to(torch.int64) * 4)

        args = self.kernel_args_template.copy()
        args.update({
            'q_ptrs': q_ptrs, 'r_ptrs': r_ptrs,
            'm_arr': device_buffers_set['q']['lengths'][:batch_size],
            'n_arr': device_buffers_set['r']['lengths'][:batch_size],
            'outs': device_buffers_set['outs'][:batch_size],
            # --- MODIFIED: Remove DP buffers ---
            # **device_buffers_set['dp'] # REMOVED - Kernel allocates H/E/F internally
            # --- END MODIFICATION ---
        })
        return args