"""
Parallel Runner - The Master Orchestrator

This is the COMMAND CENTER that:
1. Runs all 200 configurations (n=1 to n=200)
2. Selects the OPTIMAL strategy for each n
3. Executes in PARALLEL across all CPU cores
4. Checkpoints progress to avoid losing work
5. Generates the final submission

Strategy Selection:
- n=1: Exhaustive rotation search (trivial)
- n=2: Grid search + local refinement
- n=3-12: CMA-ES + Basin Hopping (heavy artillery)
- n=13-40: Adaptive SA (2M+ iterations)
- n=41-100: Hybrid (Lattice + SA refinement)
- n=101-200: Lattice + edge optimization

Parallelization:
- Uses ProcessPoolExecutor for true parallelism
- Each n is an independent job
- Checkpoints every N completions
- Can resume from checkpoint

This is where we WIN - by running everything optimally and in parallel.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import pickle
import json
import os
import time
from datetime import datetime
from tqdm import tqdm


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class OptimizationResult:
    """Result of optimizing a single configuration."""
    n: int
    solution: np.ndarray  # (n, 3) array of [x, y, angle_deg]
    score: float          # s²/n
    time_seconds: float
    method: str
    iterations: int = 0
    restarts: int = 0
    
    def to_dict(self) -> Dict:
        return {
            'n': self.n,
            'solution': self.solution.tolist(),
            'score': self.score,
            'time_seconds': self.time_seconds,
            'method': self.method,
            'iterations': self.iterations,
            'restarts': self.restarts,
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'OptimizationResult':
        return cls(
            n=d['n'],
            solution=np.array(d['solution']),
            score=d['score'],
            time_seconds=d['time_seconds'],
            method=d['method'],
            iterations=d.get('iterations', 0),
            restarts=d.get('restarts', 0),
        )


@dataclass
class RunnerConfig:
    """Configuration for the parallel runner."""
    
    # === Parallelization ===
    n_workers: int = None  # None = use all CPUs
    
    # === Strategy Thresholds ===
    cmaes_max_n: int = 12
    sa_max_n: int = 40
    hybrid_max_n: int = 100
    # n > hybrid_max_n uses lattice
    
    # === Checkpointing ===
    checkpoint_interval: int = 1  # Save after every config (safe for Colab)
    checkpoint_dir: str = "checkpoints"
    
    # === Output ===
    results_dir: str = "results"
    submission_path: str = "submission.csv"
    
    # === Strategy-specific settings ===
    cmaes_evaluations: int = 100000
    sa_iterations: int = 2000000
    hybrid_sa_iterations: int = 500000
    lattice_refinement_iters: int = 200000
    
    # === Verbosity ===
    verbose: bool = True
    progress_bar: bool = True


# =============================================================================
# STRATEGY SELECTION
# =============================================================================

def get_strategy(n: int, config: RunnerConfig) -> str:
    """Determine the optimal strategy for n trees."""
    if n == 1:
        return 'exhaustive'
    elif n == 2:
        return 'grid_search'
    elif n <= config.cmaes_max_n:
        return 'cmaes'
    elif n <= config.sa_max_n:
        return 'sa'
    elif n <= config.hybrid_max_n:
        return 'hybrid'
    else:
        return 'lattice'


def get_strategy_description(strategy: str) -> str:
    """Get human-readable description of strategy."""
    descriptions = {
        'exhaustive': 'Exhaustive rotation search',
        'grid_search': 'Grid search + refinement',
        'cmaes': 'CMA-ES + Basin Hopping',
        'sa': 'Adaptive Simulated Annealing',
        'hybrid': 'Lattice + SA refinement',
        'lattice': 'Lattice + edge optimization',
    }
    return descriptions.get(strategy, strategy)


# =============================================================================
# SINGLE CONFIGURATION OPTIMIZER
# =============================================================================

def optimize_single_n(args: Tuple) -> OptimizationResult:
    """
    Optimize a single n-tree configuration.
    
    This function is called by the process pool worker.
    
    Args:
        args: (n, strategy, config_dict)
        
    Returns:
        OptimizationResult
    """
    n, strategy, config_dict = args
    start_time = time.time()
    
    # Import inside function to avoid pickling issues
    import numpy as np
    
    try:
        if strategy == 'exhaustive':
            solution, score = _optimize_n1()
        elif strategy == 'grid_search':
            solution, score = _optimize_n2()
        elif strategy == 'cmaes':
            solution, score = _optimize_cmaes(n, config_dict)
        elif strategy == 'sa':
            solution, score = _optimize_sa(n, config_dict)
        elif strategy == 'hybrid':
            solution, score = _optimize_hybrid(n, config_dict)
        elif strategy == 'lattice':
            solution, score = _optimize_lattice(n, config_dict)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        elapsed = time.time() - start_time
        
        return OptimizationResult(
            n=n,
            solution=solution,
            score=score,
            time_seconds=elapsed,
            method=strategy,
        )
    
    except Exception as e:
        # Log the actual error for debugging
        import traceback
        print(f"ERROR in n={n} ({strategy}): {e}")
        traceback.print_exc()
        
        # Return a fallback solution
        elapsed = time.time() - start_time
        fallback = _create_fallback_solution(n)
        
        # Compute fallback score
        from core.tree_polygon import transform_trees_batch
        from core.bounding_box import compute_score
        
        positions = fallback[:, :2]
        angles = np.radians(fallback[:, 2])
        all_verts = transform_trees_batch(positions, angles)
        score = compute_score(all_verts, n)
        
        return OptimizationResult(
            n=n,
            solution=fallback,
            score=score,
            time_seconds=elapsed,
            method=f'{strategy}_fallback',
        )


def _optimize_n1():
    """Exhaustive optimization for n=1."""
    from optimizers.cmaes_optimizer import optimize_n1
    return optimize_n1()


def _optimize_n2():
    """Grid search for n=2."""
    from optimizers.cmaes_optimizer import optimize_n2
    return optimize_n2()


def _optimize_cmaes(n: int, config_dict: dict):
    """CMA-ES optimization for small n."""
    from optimizers.cmaes_optimizer import CMAESOptimizer, get_cmaes_config_for_n
    
    config = get_cmaes_config_for_n(n)
    config.max_evaluations = config_dict.get('cmaes_evaluations', 100000)
    config.verbose = False
    
    optimizer = CMAESOptimizer(config)
    return optimizer.optimize(n_trees=n, verbose=False)


def _optimize_sa(n: int, config_dict: dict):
    """Adaptive SA for medium n."""
    from optimizers.adaptive_sa import AdaptiveSA, get_config_for_n
    
    config = get_config_for_n(n)
    config.max_iterations = config_dict.get('sa_iterations', 2000000)
    
    optimizer = AdaptiveSA(config)
    return optimizer.optimize(n_trees=n, verbose=False)


def _optimize_hybrid(n: int, config_dict: dict):
    """Hybrid lattice + SA for medium-large n."""
    from packing.lattice import LatticePacker, LatticeConfig
    from optimizers.adaptive_sa import AdaptiveSA, AdaptiveSAConfig
    
    # First, get lattice solution
    lattice_config = LatticeConfig(
        use_optimized_dimer=True,
        dimer_optimization_iters=300,
        optimize_edges=True,
        edge_sa_iterations=30000,
        use_refinement=False,  # We'll do our own refinement
        verbose=False,
    )
    
    packer = LatticePacker(lattice_config)
    lattice_solution, lattice_score = packer.pack(n, verbose=False)
    
    # Then refine with SA
    sa_config = AdaptiveSAConfig(
        max_iterations=config_dict.get('hybrid_sa_iterations', 500000),
        num_restarts=5,
        initial_temp=0.5,
        initial_step_translation=0.1,
        initial_step_rotation=15.0,
    )
    
    sa = AdaptiveSA(sa_config)
    return sa.optimize(n_trees=n, initial_solution=lattice_solution, verbose=False)


def _optimize_lattice(n: int, config_dict: dict):
    """Lattice packing for large n."""
    from packing.lattice import LatticePacker, LatticeConfig
    
    config = LatticeConfig(
        use_optimized_dimer=True,
        dimer_optimization_iters=500,
        optimize_edges=True,
        edge_sa_iterations=50000,
        use_refinement=True,
        refinement_iterations=config_dict.get('lattice_refinement_iters', 200000),
        verbose=False,
    )
    
    packer = LatticePacker(config)
    return packer.pack(n, verbose=False)


def _create_fallback_solution(n: int) -> np.ndarray:
    """Create a simple fallback solution."""
    sqrt_n = int(np.ceil(np.sqrt(n)))
    spacing = 1.5
    
    solution = np.zeros((n, 3))
    idx = 0
    
    for i in range(sqrt_n):
        for j in range(sqrt_n):
            if idx >= n:
                break
            solution[idx, 0] = (i - sqrt_n/2) * spacing
            solution[idx, 1] = (j - sqrt_n/2) * spacing
            solution[idx, 2] = 0
            idx += 1
    
    return solution


# =============================================================================
# PARALLEL OPTIMIZER
# =============================================================================

class ParallelOptimizer:
    """
    Master orchestrator for optimizing all 200 configurations.
    
    Example:
        optimizer = ParallelOptimizer()
        results = optimizer.optimize_all()
        optimizer.generate_submission()
    """
    
    def __init__(self, config: RunnerConfig = None):
        self.config = config or RunnerConfig()
        
        if self.config.n_workers is None:
            self.config.n_workers = cpu_count()
        
        self.results: Dict[int, OptimizationResult] = {}
        self.start_time = None
        
        # Ensure directories exist
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        os.makedirs(self.config.results_dir, exist_ok=True)
    
    def optimize_all(
        self,
        n_range: Tuple[int, int] = (1, 200),
        resume_from: str = None
    ) -> Dict[int, OptimizationResult]:
        """
        Optimize all configurations from n_range[0] to n_range[1].
        
        Args:
            n_range: (min_n, max_n) inclusive
            resume_from: Path to checkpoint to resume from
            
        Returns:
            Dict mapping n -> OptimizationResult
        """
        cfg = self.config
        self.start_time = time.time()
        
        # Resume from checkpoint if available
        if resume_from:
            self._load_checkpoint(resume_from)
        
        # Determine which n values still need optimization
        all_n = list(range(n_range[0], n_range[1] + 1))
        pending_n = [n for n in all_n if n not in self.results]
        
        if cfg.verbose:
            print(f"\n{'='*70}")
            print(f"🎄 SANTA 2025 PARALLEL OPTIMIZER 🎄")
            print(f"{'='*70}")
            print(f"  Configurations: {n_range[0]} to {n_range[1]}")
            print(f"  Already completed: {len(self.results)}")
            print(f"  Pending: {len(pending_n)}")
            print(f"  Workers: {cfg.n_workers}")
            print(f"{'='*70}")
            
            # Show strategy distribution
            strategy_counts = {}
            for n in pending_n:
                strategy = get_strategy(n, cfg)
                strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
            
            print("\n  Strategy distribution:")
            for strategy, count in sorted(strategy_counts.items()):
                print(f"    {strategy}: {count} configurations")
        
        if not pending_n:
            print("\n  ✅ All configurations already complete!")
            return self.results
        
        # Prepare tasks
        tasks = []
        for n in pending_n:
            strategy = get_strategy(n, cfg)
            config_dict = {
                'cmaes_evaluations': cfg.cmaes_evaluations,
                'sa_iterations': cfg.sa_iterations,
                'hybrid_sa_iterations': cfg.hybrid_sa_iterations,
                'lattice_refinement_iters': cfg.lattice_refinement_iters,
            }
            tasks.append((n, strategy, config_dict))
        
        # Run in parallel
        completed = 0
        last_checkpoint = 0
        
        with ProcessPoolExecutor(max_workers=cfg.n_workers) as executor:
            # Submit all tasks
            future_to_n = {
                executor.submit(optimize_single_n, task): task[0]
                for task in tasks
            }
            
            # Process results as they complete
            if cfg.progress_bar:
                pbar = tqdm(total=len(tasks), desc="Optimizing")
            
            for future in as_completed(future_to_n):
                n = future_to_n[future]
                
                try:
                    result = future.result()
                    self.results[result.n] = result
                    
                    if cfg.verbose and not cfg.progress_bar:
                        strategy = get_strategy(n, cfg)
                        print(f"  n={n:3d} ({strategy:10s}): "
                              f"score={result.score:.6f}, "
                              f"time={result.time_seconds:.1f}s")
                    
                    completed += 1
                    
                    if cfg.progress_bar:
                        pbar.update(1)
                        pbar.set_postfix({
                            'n': n,
                            'score': f'{result.score:.4f}',
                            'total': f'{self._compute_total_score():.2f}'
                        })
                    
                    # Checkpoint
                    if completed - last_checkpoint >= cfg.checkpoint_interval:
                        self._save_checkpoint()
                        last_checkpoint = completed
                
                except Exception as e:
                    print(f"\n  ❌ Error optimizing n={n}: {e}")
            
            if cfg.progress_bar:
                pbar.close()
        
        # Final checkpoint
        self._save_checkpoint()
        
        # Summary
        elapsed = time.time() - self.start_time
        total_score = self._compute_total_score()
        
        if cfg.verbose:
            print(f"\n{'='*70}")
            print(f"  ✅ OPTIMIZATION COMPLETE!")
            print(f"     Total Score: {total_score:.6f}")
            print(f"     Time: {elapsed/60:.1f} minutes")
            print(f"     Configurations: {len(self.results)}")
            print(f"{'='*70}")
        
        return self.results
    
    def optimize_single(self, n: int) -> OptimizationResult:
        """Optimize a single configuration."""
        strategy = get_strategy(n, self.config)
        config_dict = {
            'cmaes_evaluations': self.config.cmaes_evaluations,
            'sa_iterations': self.config.sa_iterations,
            'hybrid_sa_iterations': self.config.hybrid_sa_iterations,
            'lattice_refinement_iters': self.config.lattice_refinement_iters,
        }
        
        result = optimize_single_n((n, strategy, config_dict))
        self.results[n] = result
        return result
    
    def generate_submission(self, output_path: str = None) -> str:
        """Generate competition submission file."""
        from utils.submission import generate_submission
        
        if output_path is None:
            output_path = self.config.submission_path
        
        # Convert results to solutions dict
        solutions = {n: result.solution for n, result in self.results.items()}
        
        success = generate_submission(solutions, output_path, validate=True, verbose=True)
        
        if success:
            return output_path
        else:
            raise RuntimeError("Submission generation failed validation")
    
    def get_summary(self) -> Dict:
        """Get summary statistics."""
        if not self.results:
            return {}
        
        scores = [r.score for r in self.results.values()]
        times = [r.time_seconds for r in self.results.values()]
        
        return {
            'total_score': sum(scores),
            'avg_score': np.mean(scores),
            'min_score': min(scores),
            'max_score': max(scores),
            'total_time': sum(times),
            'n_completed': len(self.results),
        }
    
    def get_worst_configurations(self, k: int = 10) -> List[Tuple[int, float]]:
        """Get the k worst-scoring configurations."""
        sorted_results = sorted(
            self.results.items(),
            key=lambda x: x[1].score,
            reverse=True
        )
        return [(n, r.score) for n, r in sorted_results[:k]]
    
    # =========================================================================
    # CHECKPOINTING
    # =========================================================================
    
    def _save_checkpoint(self):
        """Save current results to checkpoint."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as pickle for full fidelity
        pickle_path = os.path.join(
            self.config.checkpoint_dir,
            f"checkpoint_{timestamp}.pkl"
        )
        
        with open(pickle_path, 'wb') as f:
            pickle.dump(self.results, f)
        
        # Also save as JSON for human readability
        json_path = os.path.join(
            self.config.checkpoint_dir,
            f"checkpoint_{timestamp}.json"
        )
        
        json_data = {
            n: result.to_dict() for n, result in self.results.items()
        }
        
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        # Update "latest" symlink/copy
        latest_pickle = os.path.join(self.config.checkpoint_dir, "latest.pkl")
        with open(latest_pickle, 'wb') as f:
            pickle.dump(self.results, f)
    
    def _load_checkpoint(self, path: str):
        """Load results from checkpoint."""
        if path.endswith('.json'):
            with open(path, 'r') as f:
                json_data = json.load(f)
            
            self.results = {
                int(n): OptimizationResult.from_dict(d)
                for n, d in json_data.items()
            }
        else:
            with open(path, 'rb') as f:
                self.results = pickle.load(f)
        
        print(f"  Loaded {len(self.results)} results from checkpoint")
    
    def _compute_total_score(self) -> float:
        """Compute total score across all results."""
        return sum(r.score for r in self.results.values())


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def run_full_optimization(verbose: bool = True) -> Dict[int, OptimizationResult]:
    """
    Run the complete optimization pipeline.
    
    This is the main entry point for running everything.
    """
    config = RunnerConfig(
        verbose=verbose,
        progress_bar=True,
    )
    
    optimizer = ParallelOptimizer(config)
    results = optimizer.optimize_all()
    
    # Generate submission
    optimizer.generate_submission()
    
    return results


def quick_test(n_values: List[int] = None, verbose: bool = True):
    """
    Quick test with a subset of configurations.
    """
    if n_values is None:
        n_values = [1, 2, 5, 10, 20, 50, 100]
    
    config = RunnerConfig(
        cmaes_evaluations=10000,
        sa_iterations=100000,
        hybrid_sa_iterations=50000,
        lattice_refinement_iters=50000,
        verbose=verbose,
        progress_bar=True,
    )
    
    optimizer = ParallelOptimizer(config)
    
    print(f"Quick test with n = {n_values}")
    
    results = {}
    for n in n_values:
        result = optimizer.optimize_single(n)
        results[n] = result
        print(f"  n={n:3d}: score={result.score:.6f}, "
              f"method={result.method}, time={result.time_seconds:.1f}s")
    
    total = sum(r.score for r in results.values())
    print(f"\nTotal score (subset): {total:.6f}")
    
    return results


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    print("Testing Parallel Runner")
    print("=" * 70)
    
    # Warm up
    print("\nWarming up JIT compilation...")
    from core.tree_polygon import warmup as warmup_tree
    warmup_tree()
    
    # Quick test
    print("\n" + "=" * 70)
    print("Running quick test...")
    
    config = RunnerConfig(
        n_workers=4,
        cmaes_evaluations=5000,
        sa_iterations=50000,
        hybrid_sa_iterations=20000,
        lattice_refinement_iters=20000,
        verbose=True,
        progress_bar=True,
    )
    
    optimizer = ParallelOptimizer(config)
    
    # Test just a few configurations
    test_ns = [1, 3, 10, 25, 50]
    
    print(f"\nOptimizing n = {test_ns}...")
    
    for n in test_ns:
        result = optimizer.optimize_single(n)
        strategy = get_strategy(n, config)
        print(f"  n={n:3d} [{strategy:10s}]: score={result.score:.6f}, "
              f"time={result.time_seconds:.1f}s")
    
    # Summary
    summary = optimizer.get_summary()
    print(f"\n✅ Test complete!")
    print(f"   Total score (subset): {summary['total_score']:.6f}")
    print(f"   Total time: {summary['total_time']:.1f}s")
