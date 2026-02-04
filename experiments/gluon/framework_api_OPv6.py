# prototype/framework_api_OPv6.py

from typing import List, Tuple

from configs.task_config import AlignmentTask
# --- MODIFIED: Import the new Gluon kernel ---
# from core.local_dp_kernel_OPv6 import sw_kernel
from core.gluon_sw_kernel_v2 import sw_kernel_gluon # Assuming you saved it here
# --- END MODIFICATION ---
# from core.logan_kernel_v5 import logan_kernel
from host.scheduler_OPv6_2 import PipelineScheduler # Assuming scheduler_OPv6_2.py is the correct one

# ... (Default parameters remain the same) ...
BATCH_SIZE = 65536
MAX_SEQ_LEN = 10000
N_STREAMS = 2

def run_alignment_task(
    q_list: List[str],
    r_list: List[str],
    task: AlignmentTask,
    block_size: int = 256 # BLOCK for gluon kernel likely refers to vector size
) -> Tuple[List, float, float, float, float]:
    """
    Initializes and runs the appropriate alignment task on the pipeline scheduler.
    """
    cfg = task.configuration
    heuristic = cfg.pruning.heuristic

    kernel_to_run = None
    kernel_args_template = {}

    # --- 1. Kernel and Parameter Selection ---
    if heuristic == 'BANDED_Z_DROP':
        # --- MODIFIED: Select the Gluon kernel ---
        kernel_to_run = sw_kernel_gluon
        # --- END MODIFICATION ---

        band_width = cfg.pruning.params.get('band_width', 1)
        # Physical stride calculation remains the same
        PHYS_STRIDE = ((band_width + 31) // 32) * 32

        # --- MODIFIED: Remove Hbuf, Ebuf, Fbuf from template ---
        # These are no longer arguments for the Gluon kernel
        kernel_args_template = {
            # Global pointers and lengths are still needed
            # 'q_ptrs', 'r_ptrs', 'm_arr', 'n_arr', 'outs' will be added by scheduler
            # Scoring parameters are still needed
            'match_score': cfg.scoring.params['match_score'],
            'mismatch_score': cfg.scoring.params['mismatch_score'],
            'gap_open_penalty': cfg.scoring.params['gap_open_penalty'],
            'gap_extend_penalty': cfg.scoring.params['gap_extend_penalty'],
            'drop_threshold': cfg.pruning.params.get('threshold', 0),
            # Constexprs are still needed
            'BLOCK': block_size, # This is the vector size (e.g., 256)
            'SCORING_MODEL': cfg.scoring.model_type,
            'PRUNING_BAND': 'STATIC',
            'PRUNING_DROP': 'Z_DROP',
            'IS_EXTENSION': (task.align_problem == 'Extension'),
            'STRIDE': PHYS_STRIDE, # Physical stride for smem layout
            'BAND': band_width    # Logical band width
        }
        # --- END MODIFICATION ---

    # elif heuristic == 'LOGAN_X_DROP': # Keep this section if you use it
    #     kernel_to_run = logan_kernel
    #     # ... prepare kernel_args_template for logan_kernel ...

    else:
        raise NotImplementedError(f"No kernel/pipeline configured for heuristic='{heuristic}'")

    # --- 2. Initialize and Execute the Pipeline ---
    scheduler = PipelineScheduler(
        kernel_to_run=kernel_to_run,
        kernel_args_template=kernel_args_template,
        cfg=cfg,
        batch_size=BATCH_SIZE,
        max_seq_len=MAX_SEQ_LEN,
        n_streams=N_STREAMS
    )

    return scheduler.execute(q_list, r_list)