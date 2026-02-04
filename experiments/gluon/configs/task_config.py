# prototype/configs/task_config.py

from dataclasses import dataclass, field
from typing import Literal, Dict, Any

#  Dimension 4: Solution Configuration
#  This section defines the configurable parts of a chosen algorithmic solution.

@dataclass
class ScoringConfig:
    """
    Defines the Scoring Model (Sub-dimension 4A).
    - `model_type` is used as a constexpr in the kernel to switch logic.
    - `params` is a generic dictionary, allowing future extension to complex
      models (e.g., substitution matrices) without changing this class structure.
    """
    model_type: Literal['LINEAR', 'AFFINE']
    params: Dict[str, Any]

# === MAIN CHANGE: RECONSIDRE THE PRUNING CONFIGURATION ===
# Instead of seperating banding and dropping into two different classes,
# we now have a single unified heuristic.

@dataclass
class PruningConfig:
    """
    Defines the Pruning Strategy (Sub-dimension 4B).
    """

    heuristic: Literal['NONE',  # no heuristic
    'BANDED_Z_DROP', # AGAThA/extz's heuristic
    'LOGAN_X_DROP' # LOGAN's X-drop heuristic
    # Todo: Add more heuristics as needed
    ] = 'NONE'

    params: Dict[str, Any] = field(default_factory=dict)
# === CHANGE END ===

@dataclass
class SolutionConfiguration:
    """The complete configuration object for Dimension 4."""
    scoring: ScoringConfig
    pruning: PruningConfig


#  The Main AlignmentTask Object
#  This class is a direct code representation of your 4-D Design Space.

@dataclass
class AlignmentTask:
    """
    Defines a complete alignment task, directly mapping to the 4-D Design Space.
    For the MVP, D1, D2, and D3 are fixed to validate D4's flexibility.
    """
    # Dimension 4: Solution Configuration (no default value, must be first)
    configuration: SolutionConfiguration

    # Dimension 1: Application Scenario (Fixed for MVP)
    app_scenario: Literal['Pairwise', 'DatabaseSearch', 'ReadMapping'] = 'Pairwise'
    
    # Dimension 2: Alignment Problem (Selects the core algorithm)
    align_problem: Literal['Local', 'Global', 'Extension'] = 'Local'
    
    # Dimension 3: Algorithmic Solution (Selects the implementation strategy)
    algo_solution: Literal['StandardDP', 'WavefrontDP'] = 'StandardDP'


#  Configuration Helpers
#  These factory functions make it easy to create common configurations.

def create_affine_scoring(match: int, mismatch: int, open_penalty: int, extend_penalty: int) -> ScoringConfig:
    """Factory for an Affine Gap scoring configuration."""
    return ScoringConfig(
        model_type='AFFINE',
        params={
            "match_score": match, "mismatch_score": mismatch,
            "gap_open_penalty": open_penalty, "gap_extend_penalty": extend_penalty
        }
    )

def create_linear_scoring(match: int, mismatch: int, gap_penalty: int) -> ScoringConfig:
    """Factory for a Linear Gap scoring configuration."""
    # The parameters dictionary is adapted to fit the kernel's signature.
    return ScoringConfig(
        model_type='LINEAR',
        params={
            "match_score": match, "mismatch_score": mismatch,
            "gap_open_penalty": gap_penalty, "gap_extend_penalty": gap_penalty
        }
    )

def create_no_pruning() -> PruningConfig:
    """Factory for a configuration with no pruning."""
    # This now returns the new structure with the correct heuristic.
    return PruningConfig(heuristic='NONE')

# === CHANGE: new facotries that match the new unified heuristic ===
def create_banded_z_drop_pruning(band_width: int, threshold: int) -> PruningConfig:
    """Factory for AGAThA/extz's banded Z-drop heuristic."""
    return PruningConfig(
        heuristic='BANDED_Z_DROP',
        params={'band_width': band_width, 'threshold': threshold}
    )

def create_logan_x_drop_pruning(threshold: int) -> PruningConfig:
    """Factory for LOGAN's X-drop heuristic."""
    return PruningConfig(
        heuristic='LOGAN_X_DROP',
        params={'threshold': threshold}
    )

# === CHANGE END ===