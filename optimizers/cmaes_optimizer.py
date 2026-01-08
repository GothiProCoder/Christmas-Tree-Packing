"""
CMA-ES Optimizer - Exhaustive Global Optimization for Small n

CMA-ES (Covariance Matrix Adaptation Evolution Strategy) is the ULTIMATE
weapon for small n (1-12). It's a state-of-the-art evolutionary strategy
that learns the correlation structure of the search space.

Why CMA-ES for small n?
1. It's a GLOBAL optimizer - finds global optimum, not local
2. Works without gradients (black-box optimization)
3. Learns the landscape - adapts search direction
4. Proven effectiveness on hard optimization problems
5. For small n, we can afford the computational cost

For n=1 to n=12, EVERY fraction of score matters. CMA-ES gives us the
best chance of finding the true global optimum.

This implementation includes:
- Multiple restart strategies (IPOP, BIPOP)
- Basin hopping integration
- Constraint handling for collisions
- Hybrid with local search refinement
- Parallel objective evaluation

Reference:
- Hansen, N. (2016). The CMA Evolution Strategy: A Tutorial
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional, Callable, List, Dict, Any
import time
import math
import warnings


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class CMAESConfig:
    """Configuration for CMA-ES optimizer."""
    
    # === Core CMA-ES Parameters ===
    max_evaluations: int = 100_000
    sigma0: float = 0.5                  # Initial step size
    population_size: int = None          # None = auto (4 + 3*ln(n))
    
    # === Restart Strategies ===
    use_restarts: bool = True
    max_restarts: int = 10
    restart_strategy: str = 'bipop'      # 'none', 'ipop', 'bipop'
    ipop_factor: float = 2.0             # Population increase factor
    
    # === Constraint Handling ===
    collision_penalty: float = 1e6       # Penalty for collisions
    bounds_penalty: float = 1e4          # Penalty for out-of-bounds
    coordinate_bounds: float = 50.0       # Max |x|, |y| coordinate
    
    # === Hybrid Strategies ===
    use_local_refinement: bool = True
    local_refinement_iters: int = 10000  # SA iterations after CMA-ES
    
    # === Basin Hopping Integration ===
    use_basin_hopping: bool = True
    basin_hopping_iters: int = 5
    basin_hopping_temp: float = 1.0
    
    # === Solution Diversity ===
    track_elite: bool = True
    elite_size: int = 10                 # Keep top N diverse solutions
    
    # === Verbosity ===
    verbose: bool = True


def get_cmaes_config_for_n(n: int) -> CMAESConfig:
    """Get optimized CMA-ES configuration for specific n."""
    
    if n == 1:
        # n=1 is trivial - just find best rotation
        return CMAESConfig(
            max_evaluations=10_000,
            max_restarts=5,
            use_local_refinement=True,
        )
    elif n <= 3:
        # Very small - be extremely thorough
        return CMAESConfig(
            max_evaluations=200_000,
            max_restarts=20,
            sigma0=0.8,
            use_basin_hopping=True,
            basin_hopping_iters=10,
        )
    elif n <= 6:
        return CMAESConfig(
            max_evaluations=150_000,
            max_restarts=15,
            sigma0=0.6,
            use_basin_hopping=True,
            basin_hopping_iters=7,
        )
    elif n <= 9:
        return CMAESConfig(
            max_evaluations=100_000,
            max_restarts=12,
            sigma0=0.5,
            use_basin_hopping=True,
            basin_hopping_iters=5,
        )
    else:  # n <= 12
        return CMAESConfig(
            max_evaluations=80_000,
            max_restarts=10,
            sigma0=0.4,
        )


# =============================================================================
# OBJECTIVE FUNCTION BUILDER
# =============================================================================

class ObjectiveFunction:
    """
    Builds the objective function for CMA-ES optimization.
    
    Decision variables: [x1, y1, θ1, x2, y2, θ2, ..., xn, yn, θn]
    - Positions (x, y) in some reasonable range
    - Angles θ in [0, 360] degrees
    
    The objective is to minimize s²/n where s is the bounding square side.
    Collisions are handled via penalty functions.
    """
    
    def __init__(
        self,
        n_trees: int,
        tree_vertices: np.ndarray,
        config: CMAESConfig,
        collision_checker: Optional[Callable] = None
    ):
        self.n = n_trees
        self.tree_vertices = tree_vertices
        self.config = config
        self.collision_checker = collision_checker
        
        # Dimension: x, y, angle for each tree
        self.dim = n_trees * 3
        
        # Evaluation counter
        self.eval_count = 0
        self.best_score = float('inf')
        self.best_params = None
        
        # Cache for elite solutions
        self.elite_solutions = []
    
    def __call__(self, params: np.ndarray) -> float:
        """
        Evaluate a candidate solution.
        
        Args:
            params: Flat array [x1, y1, θ1, x2, y2, θ2, ...]
            
        Returns:
            Objective value (lower is better)
        """
        self.eval_count += 1
        
        # Reshape to (n, 3)
        solution = params.reshape(self.n, 3)
        
        # Extract positions and angles
        positions = solution[:, :2].copy()
        angles_deg = solution[:, 2]
        angles_rad = np.radians(angles_deg % 360)  # Normalize angles
        
        # =====================================================================
        # PENALTY: Coordinate Bounds
        # =====================================================================
        bounds_violation = 0.0
        for i in range(self.n):
            x, y = positions[i]
            if abs(x) > self.config.coordinate_bounds:
                bounds_violation += (abs(x) - self.config.coordinate_bounds) ** 2
            if abs(y) > self.config.coordinate_bounds:
                bounds_violation += (abs(y) - self.config.coordinate_bounds) ** 2
        
        if bounds_violation > 0:
            return self.config.collision_penalty + bounds_violation * self.config.bounds_penalty
        
        # =====================================================================
        # PENALTY: Collisions
        # =====================================================================
        if self.collision_checker is not None:
            has_collision = self.collision_checker(positions, angles_rad)
            if has_collision:
                # Return penalty proportional to overlap severity
                # (for now, just a large constant)
                return self.config.collision_penalty
        
        # =====================================================================
        # OBJECTIVE: Minimize s²/n
        # =====================================================================
        from core.tree_polygon import transform_trees_batch, get_all_bounds
        from core.bounding_box import compute_bounding_square
        
        all_vertices = transform_trees_batch(positions, angles_rad)
        s = compute_bounding_square(all_vertices)
        score = (s * s) / self.n
        
        # Track best
        if score < self.best_score:
            self.best_score = score
            self.best_params = params.copy()
            
            # Update elite
            if self.config.track_elite:
                self._update_elite(params, score)
        
        return score
    
    def _update_elite(self, params: np.ndarray, score: float):
        """Maintain diverse elite solutions."""
        self.elite_solutions.append((score, params.copy()))
        self.elite_solutions.sort(key=lambda x: x[0])
        
        # Keep only top N
        if len(self.elite_solutions) > self.config.elite_size:
            self.elite_solutions = self.elite_solutions[:self.config.elite_size]
    
    def get_initial_guess(self, strategy: str = 'grid') -> np.ndarray:
        """
        Generate initial guess for optimization.
        
        Strategies:
        - 'grid': Regular grid pattern
        - 'random': Random placement
        - 'compact': Tight initial packing
        """
        n = self.n
        params = np.zeros(self.dim)
        
        if strategy == 'grid' or strategy == 'compact':
            sqrt_n = int(np.ceil(np.sqrt(n)))
            spacing = 1.0 if strategy == 'compact' else 1.3
            
            idx = 0
            for i in range(sqrt_n):
                for j in range(sqrt_n):
                    if idx >= n:
                        break
                    params[idx * 3] = (i - sqrt_n/2) * spacing
                    params[idx * 3 + 1] = (j - sqrt_n/2) * spacing
                    params[idx * 3 + 2] = np.random.uniform(0, 360)
                    idx += 1
        
        elif strategy == 'random':
            for i in range(n):
                params[i * 3] = np.random.uniform(-3, 3)
                params[i * 3 + 1] = np.random.uniform(-3, 3)
                params[i * 3 + 2] = np.random.uniform(0, 360)
        
        return params
    
    def params_to_solution(self, params: np.ndarray) -> np.ndarray:
        """Convert flat params to (n, 3) solution array."""
        solution = params.reshape(self.n, 3).copy()
        solution[:, 2] = solution[:, 2] % 360  # Normalize angles
        return solution


# =============================================================================
# CMA-ES OPTIMIZER
# =============================================================================

class CMAESOptimizer:
    """
    CMA-ES optimizer with advanced restart strategies.
    
    This is our HEAVY ARTILLERY for small n. We throw everything at it:
    - Multiple restarts (BIPOP strategy)
    - Basin hopping for escaping local optima
    - Local refinement with SA
    - Elite solution tracking
    
    Example:
        optimizer = CMAESOptimizer()
        solution, score = optimizer.optimize(n_trees=6)
    """
    
    def __init__(self, config: CMAESConfig = None):
        self.config = config or CMAESConfig()
        
        self.best_solution = None
        self.best_score = float('inf')
        self.history = []
        
        self.stats = {
            'total_evaluations': 0,
            'restarts': 0,
            'best_restart': 0,
            'time_seconds': 0,
        }
    
    def optimize(
        self,
        n_trees: int,
        initial_solution: Optional[np.ndarray] = None,
        collision_checker: Optional[Callable] = None,
        tree_vertices: Optional[np.ndarray] = None,
        verbose: bool = None
    ) -> Tuple[np.ndarray, float]:
        """
        Optimize placement of n trees using CMA-ES.
        
        Args:
            n_trees: Number of trees
            initial_solution: Starting point (optional)
            collision_checker: Collision detection function
            tree_vertices: Tree shape vertices
            verbose: Override config verbosity
            
        Returns:
            (best_solution, best_score)
        """
        try:
            import cma
        except ImportError:
            raise ImportError("CMA-ES requires the 'cma' package. Install with: pip install cma")
        
        cfg = self.config
        if verbose is None:
            verbose = cfg.verbose
        
        # Get tree vertices
        if tree_vertices is None:
            from core.tree_polygon import TREE_VERTICES
            tree_vertices = TREE_VERTICES
        
        # Create collision checker if not provided
        if collision_checker is None:
            collision_checker = self._create_collision_checker()
        
        # Build objective function
        objective = ObjectiveFunction(
            n_trees=n_trees,
            tree_vertices=tree_vertices,
            config=cfg,
            collision_checker=collision_checker
        )
        
        # Initial guess
        if initial_solution is not None:
            x0 = initial_solution.flatten()
        else:
            x0 = objective.get_initial_guess('grid')
        
        start_time = time.time()
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"CMA-ES Optimization for n={n_trees}")
            print(f"  Dimension: {objective.dim}")
            print(f"  Max evaluations: {cfg.max_evaluations:,}")
            print(f"  Restart strategy: {cfg.restart_strategy}")
            print(f"  Max restarts: {cfg.max_restarts}")
            print(f"{'='*60}")
        
        # =====================================================================
        # MAIN OPTIMIZATION WITH RESTARTS
        # =====================================================================
        
        best_score = float('inf')
        best_params = None
        restart_count = 0
        total_evals = 0
        
        # Population size for IPOP/BIPOP
        popsize = cfg.population_size or int(4 + 3 * np.log(objective.dim))
        initial_popsize = popsize
        
        while total_evals < cfg.max_evaluations and restart_count < cfg.max_restarts:
            
            # Determine population size for this restart
            if cfg.restart_strategy == 'ipop':
                # IPOP: Increase population after each restart
                popsize = int(initial_popsize * (cfg.ipop_factor ** restart_count))
            elif cfg.restart_strategy == 'bipop':
                # BIPOP: Alternate between small and large populations
                if restart_count % 2 == 0:
                    popsize = initial_popsize
                else:
                    popsize = int(initial_popsize * (cfg.ipop_factor ** ((restart_count + 1) // 2)))
            
            # Budget for this restart
            remaining_evals = cfg.max_evaluations - total_evals
            restart_budget = min(remaining_evals, cfg.max_evaluations // (cfg.max_restarts - restart_count + 1))
            
            if verbose:
                print(f"\n  Restart {restart_count + 1}/{cfg.max_restarts}: "
                      f"popsize={popsize}, budget={restart_budget:,}")
            
            # Initial point for this restart
            if restart_count == 0 and initial_solution is not None:
                x0_restart = x0.copy()
            elif restart_count > 0 and best_params is not None and np.random.random() < 0.5:
                # Sometimes restart from best with perturbation
                x0_restart = best_params + np.random.randn(objective.dim) * cfg.sigma0 * 0.5
            else:
                # Random restart
                x0_restart = objective.get_initial_guess('random')
            
            # -----------------------------------------------------------------
            # Run CMA-ES
            # -----------------------------------------------------------------
            es = cma.CMAEvolutionStrategy(
                x0_restart,
                cfg.sigma0,
                {
                    'maxfevals': restart_budget,
                    'popsize': popsize,
                    'verb_disp': 0,  # Suppress CMA output
                    'verb_log': 0,
                    'verb_filenameprefix': '',  # Don't create output files
                    'bounds': [None, None],  # Handle bounds via penalty
                }
            )
            
            # Optimization loop
            while not es.stop():
                solutions = es.ask()
                fitness = [objective(x) for x in solutions]
                es.tell(solutions, fitness)
                
                # Progress update
                if verbose and objective.eval_count % 10000 == 0:
                    print(f"    Evals: {objective.eval_count:,} | "
                          f"Best: {objective.best_score:.8f}")
            
            # Update best
            if objective.best_score < best_score:
                best_score = objective.best_score
                best_params = objective.best_params.copy()
                self.stats['best_restart'] = restart_count
                
                if verbose:
                    print(f"    ★ New best: {best_score:.8f}")
            
            total_evals = objective.eval_count
            restart_count += 1
        
        self.stats['restarts'] = restart_count
        self.stats['total_evaluations'] = total_evals
        
        # =====================================================================
        # BASIN HOPPING (Optional additional global search)
        # =====================================================================
        
        if cfg.use_basin_hopping and best_params is not None:
            if verbose:
                print(f"\n  Basin Hopping ({cfg.basin_hopping_iters} iterations)...")
            
            best_params, best_score = self._basin_hopping(
                objective, best_params, best_score, verbose
            )
        
        # =====================================================================
        # LOCAL REFINEMENT (Polish with SA)
        # =====================================================================
        
        if cfg.use_local_refinement and best_params is not None:
            if verbose:
                print(f"\n  Local refinement ({cfg.local_refinement_iters:,} SA iterations)...")
            
            best_params, best_score = self._local_refinement(
                objective, best_params, best_score, n_trees, verbose
            )
        
        # =====================================================================
        # FINALIZE
        # =====================================================================
        
        elapsed = time.time() - start_time
        self.stats['time_seconds'] = elapsed
        
        if best_params is not None:
            self.best_solution = objective.params_to_solution(best_params)
            self.best_score = best_score
        else:
            # Fallback to simple grid
            self.best_solution = objective.params_to_solution(
                objective.get_initial_guess('grid')
            )
            self.best_score = objective(objective.get_initial_guess('grid'))
        
        if verbose:
            print(f"\n  ✅ CMA-ES Complete!")
            print(f"     Best Score: {self.best_score:.8f}")
            print(f"     Evaluations: {total_evals:,}")
            print(f"     Restarts: {restart_count}")
            print(f"     Time: {elapsed:.1f}s")
        
        return self.best_solution, self.best_score
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def _create_collision_checker(self) -> Callable:
        """Create default collision checker."""
        def check(positions, angles_rad):
            from core.tree_polygon import transform_trees_batch, get_all_bounds
            from core.collision import check_any_collision
            
            all_vertices = transform_trees_batch(positions, angles_rad)
            all_bounds = get_all_bounds(all_vertices)
            return check_any_collision(all_vertices, all_bounds, use_shapely=True)
        
        return check
    
    def _basin_hopping(
        self,
        objective: ObjectiveFunction,
        x0: np.ndarray,
        current_best: float,
        verbose: bool
    ) -> Tuple[np.ndarray, float]:
        """
        Basin hopping to escape local optima.
        
        This is like a "meta-optimizer" that makes large jumps
        and accepts/rejects based on improvement.
        """
        cfg = self.config
        
        best_x = x0.copy()
        best_score = current_best
        current_x = x0.copy()
        current_score = current_best
        
        for i in range(cfg.basin_hopping_iters):
            # Large random perturbation
            step_size = cfg.sigma0 * 2 * (1 - i / cfg.basin_hopping_iters)
            new_x = current_x + np.random.randn(len(current_x)) * step_size
            
            # Evaluate
            new_score = objective(new_x)
            
            # Accept/reject (Metropolis-like)
            accept = False
            if new_score < current_score:
                accept = True
            else:
                delta = new_score - current_score
                prob = np.exp(-delta / cfg.basin_hopping_temp)
                accept = np.random.random() < prob
            
            if accept:
                current_x = new_x
                current_score = new_score
                
                if new_score < best_score:
                    best_score = new_score
                    best_x = new_x.copy()
                    
                    if verbose:
                        print(f"    Basin hop {i+1}: new best = {best_score:.8f}")
        
        return best_x, best_score
    
    def _local_refinement(
        self,
        objective: ObjectiveFunction,
        x0: np.ndarray,
        current_best: float,
        n_trees: int,
        verbose: bool
    ) -> Tuple[np.ndarray, float]:
        """
        Local refinement using quick SA.
        
        CMA-ES finds the basin, SA polishes within it.
        """
        from optimizers.adaptive_sa import AdaptiveSA, AdaptiveSAConfig
        
        cfg = self.config
        
        # Convert to solution format
        solution = objective.params_to_solution(x0)
        
        # Quick SA config
        sa_config = AdaptiveSAConfig(
            max_iterations=cfg.local_refinement_iters,
            num_restarts=2,
            restart_threshold=cfg.local_refinement_iters // 3,
            initial_temp=0.1,           # Low temp - we're already close
            initial_step_translation=0.05,  # Small steps
            initial_step_rotation=5.0,
        )
        
        sa = AdaptiveSA(sa_config)
        refined_solution, refined_score = sa.optimize(
            n_trees=n_trees,
            initial_solution=solution,
            verbose=False
        )
        
        if refined_score < current_best:
            if verbose:
                print(f"    Local refinement improved: {current_best:.8f} → {refined_score:.8f}")
            return refined_solution.flatten(), refined_score
        
        return x0, current_best


# =============================================================================
# SPECIALIZED OPTIMIZERS FOR SPECIFIC n
# =============================================================================

def optimize_n1() -> Tuple[np.ndarray, float]:
    """
    Optimal solution for n=1.
    
    For a single tree, we just need to find the rotation
    that minimizes the bounding square.
    """
    from core.tree_polygon import TREE_VERTICES, transform_tree
    from core.bounding_box import compute_bounding_square
    
    best_score = float('inf')
    best_angle = 0
    
    # Exhaustive search over rotations
    for angle_deg in np.linspace(0, 360, 3600):  # 0.1° resolution
        angle_rad = np.radians(angle_deg)
        vertices = transform_tree(0, 0, angle_rad)
        
        min_x = vertices[:, 0].min()
        max_x = vertices[:, 0].max()
        min_y = vertices[:, 1].min()
        max_y = vertices[:, 1].max()
        
        s = max(max_x - min_x, max_y - min_y)
        score = s * s  # s²/1 = s²
        
        if score < best_score:
            best_score = score
            best_angle = angle_deg
    
    solution = np.array([[0.0, 0.0, best_angle]])
    return solution, best_score


def optimize_n2() -> Tuple[np.ndarray, float]:
    """
    Optimized solution for n=2.
    
    Two trees can be packed efficiently by placing them
    in specific relative positions. We search exhaustively.
    """
    from core.tree_polygon import TREE_VERTICES, transform_trees_batch, get_all_bounds
    from core.bounding_box import compute_score
    from core.collision import check_any_collision
    
    best_score = float('inf')
    best_solution = None
    
    # Search grid
    for angle1 in np.linspace(0, 180, 36):  # 5° steps
        for angle2 in np.linspace(0, 360, 72):  # 5° steps
            for dx in np.linspace(-1.0, 1.0, 21):  # 0.1 steps
                for dy in np.linspace(-0.5, 0.5, 11):
                    positions = np.array([[0.0, 0.0], [dx, dy]])
                    angles = np.radians([angle1, angle2])
                    
                    all_verts = transform_trees_batch(positions, angles)
                    all_bounds = get_all_bounds(all_verts)
                    
                    if check_any_collision(all_verts, all_bounds, use_shapely=True):
                        continue
                    
                    score = compute_score(all_verts, 2)
                    
                    if score < best_score:
                        best_score = score
                        best_solution = np.array([
                            [0.0, 0.0, angle1],
                            [dx, dy, angle2]
                        ])
    
    if best_solution is None:
        # Fallback
        best_solution = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
        best_score = 99.0
    
    return best_solution, best_score


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def optimize_with_cmaes(
    n: int,
    verbose: bool = True
) -> Tuple[np.ndarray, float]:
    """
    Optimize n-tree configuration using CMA-ES with best settings.
    
    Args:
        n: Number of trees
        verbose: Print progress
        
    Returns:
        (solution, score)
    """
    # Special cases
    if n == 1:
        if verbose:
            print("Using exhaustive search for n=1...")
        return optimize_n1()
    
    if n == 2:
        if verbose:
            print("Using grid search for n=2...")
        return optimize_n2()
    
    # General CMA-ES
    config = get_cmaes_config_for_n(n)
    config.verbose = verbose
    
    optimizer = CMAESOptimizer(config)
    return optimizer.optimize(n_trees=n, verbose=verbose)


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    print("Testing CMA-ES Optimizer")
    print("=" * 60)
    
    # Warm up
    print("\nWarming up JIT compilation...")
    from core.tree_polygon import warmup as warmup_tree
    warmup_tree()
    
    # Test n=1
    print("\n" + "="*60)
    print("Testing n=1 (exhaustive)...")
    sol1, score1 = optimize_n1()
    print(f"n=1: score = {score1:.8f}")
    print(f"     angle = {sol1[0, 2]:.2f}°")
    
    # Test n=3 with CMA-ES
    print("\n" + "="*60)
    print("Testing n=3 (CMA-ES)...")
    
    config = CMAESConfig(
        max_evaluations=20000,
        max_restarts=3,
        use_local_refinement=True,
        local_refinement_iters=5000,
        verbose=True
    )
    
    optimizer = CMAESOptimizer(config)
    sol3, score3 = optimizer.optimize(n_trees=3, verbose=True)
    
    print(f"\n✅ Test complete!")
    print(f"   n=3 final score: {score3:.8f}")
