"""
Runners module - Parallel optimization and solution management.
"""

from .parallel_runner import ParallelOptimizer, OptimizationResult
from .solution_manager import SolutionManager

__all__ = [
    'ParallelOptimizer',
    'OptimizationResult',
    'SolutionManager',
]
