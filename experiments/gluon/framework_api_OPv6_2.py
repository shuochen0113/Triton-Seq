# prototype/framework_api_OPv6.py

from typing import List, Tuple

from configs.task_config import AlignmentTask
# --- MODIFIED: Import the new Gluon kernel ---
# from core.local_dp_kernel_OPv6 import sw_kernel
# from core.gluon_sw_kernel_v2 import sw_kernel_gluon # Assuming you saved it here
from core.gluon_sw_kernel_v3 import sw_kernel_gluon_1x1desc # New kernel
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
    block_size: int = 256
) -> Tuple[List, float, float, float, float]:
    """
    Initializes and runs the appropriate alignment task on the pipeline scheduler.
    """
    cfg = task.configuration
    heuristic = cfg.pruning.heuristic

    kernel_to_run = None
    kernel_args_template = {}

    if heuristic == 'BANDED_Z_DROP':
        # --- MODIFIED: Select the new kernel ---
        kernel_to_run = sw_kernel_gluon_1x1desc
        # --- END MODIFICATION ---

        band_width = cfg.pruning.params.get('band_width', 1)
        # Original stride calculation
        ORIG_STRIDE = ((band_width + 31) // 32) * 32

        # --- MODIFIED: Define power-of-2 parameters for the kernel ---
        # Calculate next power of 2 for STRIDE (e.g., 768 -> 1024)
        STRIDE_POW2 = 1 << (ORIG_STRIDE - 1).bit_length() if ORIG_STRIDE > 0 else 1
        H_SLOTS = 4 # Use 4 instead of 3
        EF_SLOTS = 2 # 2 is already power of 2
        # --- END MODIFICATION ---

        kernel_args_template = {
            'match_score': cfg.scoring.params['match_score'],
            'mismatch_score': cfg.scoring.params['mismatch_score'],
            'gap_open_penalty': cfg.scoring.params['gap_open_penalty'],
            'gap_extend_penalty': cfg.scoring.params['gap_extend_penalty'],
            'drop_threshold': cfg.pruning.params.get('threshold', 0),
            'BLOCK': block_size,
            'SCORING_MODEL': cfg.scoring.model_type,
            'PRUNING_BAND': 'STATIC',
            'PRUNING_DROP': 'Z_DROP',
            'IS_EXTENSION': (task.align_problem == 'Extension'),
            # --- MODIFIED: Pass new constexprs ---
            'STRIDE_POW2': STRIDE_POW2,
            'H_SLOTS': H_SLOTS,
            'EF_SLOTS': EF_SLOTS,
            # --- END MODIFICATION ---
            'BAND': band_width # Logical band width remains the same
            # REMOVED: 'STRIDE': ORIG_STRIDE (replaced by STRIDE_POW2 for allocation)
        }
        # --- END MODIFICATION ---

    # elif heuristic == 'LOGAN_X_DROP': # Keep this section if you use it
    #     kernel_to_run = logan_kernel
    #     # ... prepare kernel_args_template for logan_kernel ...

    else:
        raise NotImplementedError(f"No kernel/pipeline configured for heuristic='{heuristic}'")

    # --- Initialize and Execute Pipeline (remains the same) ---
    scheduler = PipelineScheduler(
        kernel_to_run=kernel_to_run,
        kernel_args_template=kernel_args_template,
        cfg=cfg,
        batch_size=BATCH_SIZE,
        max_seq_len=MAX_SEQ_LEN,
        n_streams=N_STREAMS
    )

    # Make sure scheduler passes base pointers correctly if needed by Gluon kernel
    # (Current scheduler logic seems okay, passing data_ptr results)
    return scheduler.execute(q_list, r_list)