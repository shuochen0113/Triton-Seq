# prototype/framework_api_OPv6.py
"""
High-level public API for the sequence alignment framework.
This module selects the appropriate kernel and dispatches the task
to the high-performance pipeline scheduler.
"""
from typing import List, Tuple

from configs.task_config import AlignmentTask
from core.local_dp_kernel_OPv6 import sw_kernel
from core.logan_kernel_v5 import logan_kernel # Assuming this is also adapted
from host.scheduler_OPv6_2 import PipelineScheduler

# --- Default pipeline parameters ---
# These can be overridden or configured as needed

# NOTE: we can combine it with triton's auto-tune ability (TODO)
BATCH_SIZE = 65536 # 65536 is the optimal batch size for RTX 4090

# NOTE: this is the maximum sequence length that can be processed in one batch
MAX_SEQ_LEN = 10000

N_STREAMS = 2
# --- end of default parameters ---

def run_alignment_task(
    q_list: List[str], 
    r_list: List[str],
    task: AlignmentTask,
    block_size: int = 256
) -> Tuple[List, float, float, float, float]:
    """
    Initializes and runs the appropriate alignment task on the pipeline scheduler.
    """
    cfg = task.configuration
    heuristic = cfg.pruning.heuristic

    kernel_to_run = None
    kernel_args_template = {}
    
    # --- 1. Kernel and Parameter Selection ---
    # This block determines WHICH kernel to run and WHAT its static parameters are.
    if heuristic == 'BANDED_Z_DROP':
        kernel_to_run = sw_kernel
        band_width = cfg.pruning.params.get('band_width', 1)
        # Physical stride in int32 elements (round up to 128B = 32*int32 alignment)
        PHYS_STRIDE = ((band_width + 31) // 32) * 32

        # Select OPv6 Smith-Waterman kernel
        kernel_to_run = sw_kernel

        kernel_args_template = {
            'match_score': cfg.scoring.params['match_score'],
            'mismatch_score': cfg.scoring.params['mismatch_score'],
            'gap_open_penalty': cfg.scoring.params['gap_open_penalty'],
            'gap_extend_penalty': cfg.scoring.params['gap_extend_penalty'],
            'drop_threshold': cfg.pruning.params.get('threshold', 0),
            'BLOCK': block_size,
            'SCORING_MODEL': cfg.scoring.model_type,
            'PRUNING_BAND': 'STATIC', 'PRUNING_DROP': 'Z_DROP',
            'IS_EXTENSION': (task.align_problem == 'Extension'),
            # NOTE: STRIDE is the physical, padded stride (>= BAND) to preserve 128B coalescing
            'STRIDE': PHYS_STRIDE,
            'BAND': band_width
        }

    elif heuristic == 'LOGAN_X_DROP':
        kernel_to_run = logan_kernel
        # ... prepare kernel_args_template for logan_kernel ...
        
    else:
        raise NotImplementedError(f"No kernel/pipeline configured for heuristic='{heuristic}'")

    # --- 2. Initialize and Execute the Pipeline ---
    # The scheduler is instantiated with the "what" (kernel & params)
    scheduler = PipelineScheduler(
        kernel_to_run=kernel_to_run,
        kernel_args_template=kernel_args_template,
        cfg=cfg,
        batch_size=BATCH_SIZE,
        max_seq_len=MAX_SEQ_LEN,
        n_streams=N_STREAMS
    )
    
    # And then told to process the "data" (q_list, r_list)
    return scheduler.execute(q_list, r_list)