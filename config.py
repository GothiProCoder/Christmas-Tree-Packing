"""
Santa 2025 - Global Configuration
All hyperparameters and settings in one place.
"""

from dataclasses import dataclass, field
from typing import Dict, Any
import os

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
SUBMISSIONS_DIR = os.path.join(PROJECT_ROOT, "submissions")
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "checkpoints")

# Ensure directories exist
for d in [RESULTS_DIR, SUBMISSIONS_DIR, CHECKPOINTS_DIR]:
    os.makedirs(d, exist_ok=True)


@dataclass
class TreeGeometry:
    """Christmas tree polygon specifications."""
    # Trunk dimensions
    trunk_width: float = 0.15
    trunk_height: float = 0.2
    
    # Tier widths
    base_width: float = 0.7
    mid_width: float = 0.4
    top_width: float = 0.25
    
    # Tip position
    tip_y: float = 0.8


@dataclass 
class SAConfig:
    """Simulated Annealing configuration."""
    # Iteration settings
    max_iterations: int = 2_000_000
    
    # Temperature schedule (Modified Lam)
    initial_temp: float = 10.0
    min_temp: float = 1e-8
    target_accept_rate: float = 0.44
    lam_rate: float = 0.99
    
    # Move parameters
    initial_step_translation: float = 0.5
    initial_step_rotation: float = 45.0  # degrees
    min_step_translation: float = 0.0001
    min_step_rotation: float = 0.01
    
    # Multi-scale moves
    use_multi_scale: bool = True
    large_move_prob: float = 0.1
    medium_move_prob: float = 0.3
    
    # Restart settings
    num_restarts: int = 10
    restart_threshold: int = 50000


@dataclass
class CMAESConfig:
    """CMA-ES configuration."""
    max_iterations: int = 20000
    sigma0: float = 0.5
    population_size: int = None  # Auto-determined
    collision_penalty: float = 1e6


@dataclass
class LatticeConfig:
    """Lattice packing configuration."""
    # Dimer optimization
    dimer_optimization_iterations: int = 10000
    
    # Grid spacing buffer
    spacing_buffer: float = 0.01


@dataclass
class ParallelConfig:
    """Parallel processing configuration."""
    num_workers: int = None  # None = auto (cpu_count)
    checkpoint_interval: int = 10
    verbose: bool = True


@dataclass
class StrategyThresholds:
    """Thresholds for strategy selection based on n."""
    cmaes_max_n: int = 12      # Use CMA-ES for n <= 12
    sa_max_n: int = 40         # Use pure SA for n <= 40
    hybrid_max_n: int = 100    # Use hybrid for n <= 100
    # n > 100 uses pure lattice with refinement


@dataclass
class OptimizationConfig:
    """Master configuration combining all sub-configs."""
    tree: TreeGeometry = field(default_factory=TreeGeometry)
    sa: SAConfig = field(default_factory=SAConfig)
    cmaes: CMAESConfig = field(default_factory=CMAESConfig)
    lattice: LatticeConfig = field(default_factory=LatticeConfig)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    thresholds: StrategyThresholds = field(default_factory=StrategyThresholds)


# Global configuration instance
CONFIG = OptimizationConfig()


def get_strategy_for_n(n: int) -> str:
    """Determine optimal strategy based on n."""
    if n <= CONFIG.thresholds.cmaes_max_n:
        return 'cmaes'
    elif n <= CONFIG.thresholds.sa_max_n:
        return 'sa'
    elif n <= CONFIG.thresholds.hybrid_max_n:
        return 'hybrid'
    else:
        return 'lattice'


def get_iterations_for_n(n: int) -> int:
    """Get appropriate iteration count based on n."""
    strategy = get_strategy_for_n(n)
    
    if strategy == 'cmaes':
        return CONFIG.cmaes.max_iterations
    elif strategy == 'sa':
        return CONFIG.sa.max_iterations
    elif strategy == 'hybrid':
        return CONFIG.sa.max_iterations // 4  # Refinement iterations
    else:
        return CONFIG.sa.max_iterations // 10  # Light refinement
