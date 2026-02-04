# prototype/host/scheduler_OPv6.py (FIXED)
"""
Implements a high-performance, asynchronous, double-buffered pipeline scheduler.
This class manages CUDA streams, buffer cycling, and overlaps CPU/GPU work.
"""
import torch
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

    def execute(self, q_list: List[str], r_list: List[str]):
        n_total_pairs = len(q_list)
        if n_total_pairs == 0:
            return [], 0.0, 0.0, 0.0, 0.0
        
        # --- 1. Initialize Pipeline Resources ---
        streams = [torch.cuda.Stream() for _ in range(self.n_streams)]
        kernel_name = getattr(self.kernel_to_run, "__name__", "triton_kernel")
        host_buffers, device_buffers = create_pipeline_resources(
            kernel_name, self.cfg, self.n_streams, self.batch_size, self.max_seq_len
        )        
        all_results = []
        batch_sizes = []
        events = {'h2d': [], 'pack': [], 'dp_init': [], 'align': [], 'd2h': []}

        # --- 2. Main Asynchronous Pipeline Loop ---
        num_batches = (n_total_pairs + self.batch_size - 1) // self.batch_size

        for i in range(num_batches):
            stream_idx = i % self.n_streams
            current_stream = streams[stream_idx]
            
            # A. Sync and collect results from this stream's previous run
            current_stream.synchronize()
            if i >= self.n_streams:
                prev_batch_idx = i - self.n_streams
                prev_batch_size = batch_sizes[prev_batch_idx]
                results_cpu = host_buffers[stream_idx]['results'][:prev_batch_size].tolist()
                all_results.extend([(int(s), (int(ii), int(j))) for s, ii, j in results_cpu])

            # B. Prepare current batch on CPU
            start, end = i * self.batch_size, min((i + 1) * self.batch_size, n_total_pairs)
            q_batch, r_batch = q_list[start:end], r_list[start:end]
            current_batch_size = len(q_batch)
            batch_sizes.append(current_batch_size)
            
            # C. Enqueue all GPU work on the current stream
            with torch.cuda.stream(current_stream):
                # C.1 Pack and H2D
                pack_events = {}
                pack_batch_into_buffers(q_batch, device_buffers[stream_idx]['q'], pack_events)
                events['h2d'].append(pack_events['h2d'])
                events['pack'].append(pack_events['pack_kernel'])
                
                pack_events = {}
                pack_batch_into_buffers(r_batch, device_buffers[stream_idx]['r'], pack_events)
                events['h2d'].append(pack_events['h2d'])
                events['pack'].append(pack_events['pack_kernel'])
                
                # C.2 DP Buffer Init
                dp_init_start = torch.cuda.Event(enable_timing=True)
                dp_init_end = torch.cuda.Event(enable_timing=True)
                dp_init_start.record()
                # =================== FIX 09/30 ===================
                # The correct initial value is -10,000,000, not 0.
                neg_inf32 = -10_000_000
                for buf in device_buffers[stream_idx]['dp'].values():
                    if isinstance(buf, torch.Tensor):
                        buf.fill_(neg_inf32)
                # ===============================================================
                dp_init_end.record()
                events['dp_init'].append((dp_init_start, dp_init_end))
                
                # C.3 Align Kernel
                align_start = torch.cuda.Event(enable_timing=True)
                align_end = torch.cuda.Event(enable_timing=True)
                align_start.record()
                
                current_kernel_args = self._prepare_kernel_args(device_buffers[stream_idx], current_batch_size)
                self.kernel_to_run[current_batch_size,](**current_kernel_args)

                align_end.record()
                events['align'].append((align_start, align_end))

                # C.4 D2H Results
                d2h_start = torch.cuda.Event(enable_timing=True)
                d2h_end = torch.cuda.Event(enable_timing=True)
                d2h_start.record()
                host_buffers[stream_idx]['results'][:current_batch_size].copy_(
                    device_buffers[stream_idx]['outs'][:current_batch_size], non_blocking=True
                )
                d2h_end.record()
                events['d2h'].append((d2h_start, d2h_end))

        # --- 3. Finalization ---
        num_collected_in_loop = max(0, num_batches - self.n_streams)
        
        for i in range(num_collected_in_loop, num_batches):
            stream_idx = i % self.n_streams
            current_stream = streams[stream_idx]
            current_stream.synchronize()
            
            batch_size = batch_sizes[i]
            results_cpu = host_buffers[stream_idx]['results'][:batch_size].tolist()
            all_results.extend([(int(s), (int(ii), int(j))) for s, ii, j in results_cpu])

        # --- 4. Timing Aggregation ---
        total_kernel_time_ms = sum(s.elapsed_time(e) for s, e in events['align'])
        packing_kernel_time_ms = sum(s.elapsed_time(e) for s, e in events['pack'])
        total_d2h_time_ms = sum(s.elapsed_time(e) for s, e in events['d2h'])
        total_h2d_time_ms = sum(s.elapsed_time(e) for s, e in events['h2d'])
        total_dp_init_time_ms = sum(s.elapsed_time(e) for s, e in events['dp_init'])
        total_setup_ms = total_h2d_time_ms + packing_kernel_time_ms + total_dp_init_time_ms
        
        return all_results, total_kernel_time_ms, total_setup_ms, packing_kernel_time_ms, total_d2h_time_ms

    def _prepare_kernel_args(self, device_buffers_set, batch_size):
        # Helper to construct the args dict for the kernel for the current batch
        q_ptrs = device_buffers_set['q']['packed_u32'].data_ptr() + (device_buffers_set['q']['offsets'][:batch_size].to(torch.int64) * 4)
        r_ptrs = device_buffers_set['r']['packed_u32'].data_ptr() + (device_buffers_set['r']['offsets'][:batch_size].to(torch.int64) * 4)
        
        args = self.kernel_args_template.copy()
        args.update({
            'q_ptrs': q_ptrs, 'r_ptrs': r_ptrs,
            'm_arr': device_buffers_set['q']['lengths'][:batch_size],
            'n_arr': device_buffers_set['r']['lengths'][:batch_size],
            'outs': device_buffers_set['outs'][:batch_size],
            **device_buffers_set['dp'] # Unpack Hbuf, Ebuf, etc.
        })
        return args