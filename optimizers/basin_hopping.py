"""
Basin Hopping Optimizer - Global Optimization via Iterative Local Search

Basin Hopping is a POWERFUL global optimization technique that:
1. Makes large random jumps to explore different "basins"
2. Performs local optimization within each basin
3. Accepts/rejects basins using Metropolis criterion

Think of it like climbing mountains:
- Local optimizers get stuck at the top of the nearest hill
- Basin Hopping JUMPS to different hills, finds the highest one

This is CRITICAL for:
- Escaping deep local minima
- Finding globally optimal tree arrangements
- Combining with SA or CMA-ES for best results

Key innovations in this implementation:
1. Adaptive step sizes - learns optimal jump distance
2. Monotonic basin hopping - always accept improvements
3. History-guided jumps - avoid revisiting same basins
4. Parallel basin exploration
5. Solution diversity maintenance

Reference:
- Wales & Doye (1997). "Global Optimization by Basin-Hopping"
- scipy.optimize.basinhopping documentation
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional, Callable, List, Dict, Any
import time
import math
from concurrent.futures import ProcessPoolExecutor, as_completed


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class BasinHoppingConfig:
    """Configuration for Basin Hopping optimizer."""
    
    # === Core Parameters ===
    n_iterations: int = 100              # Number of basin hops
    temperature: float = 1.0             # Metropolis temperature
    
    # === Step Parameters ===
    initial_step_size: float = 0.5       # Initial jump size
    step_size_position: float = 0.5      # Step for position jumps
    step_size_angle: float = 30.0        # Step for angle jumps (degrees)
    adaptive_step: bool = True           # Adapt step size
    target_accept_rate: float = 0.5      # Target acceptance rate
    step_adaptation_rate: float = 1.2    # How fast to adapt
    
    # === Local Minimization ===
    local_minimizer: str = 'sa'          # 'sa', 'nelder_mead', 'powell', 'bfgs'
    local_iterations: int = 10000        # Iterations for local minimizer
    
    # === Monotonic Mode ===
    monotonic: bool = False              # If True, only accept improvements
    
    # === History & Diversity ===
    use_history: bool = True             # Track visited basins
    history_distance_threshold: float = 0.1  # Min distance to consider "new"
    tabu_tenure: int = 10                # How long to avoid recent basins
    
    # === Parallelization ===
    n_parallel_basins: int = 1           # Parallel basin explorations
    
    # === Restart Strategy ===
    n_restarts: int = 5                  # Number of full restarts
    restart_from_best: float = 0.5       # Probability to restart from best
    
    # === Bounds ===
    coordinate_bounds: float = 50.0
    
    # === Verbosity ===
    verbose: bool = True


def get_basin_hopping_config_for_n(n: int) -> BasinHoppingConfig:
    """Get optimized configuration for specific n."""
    
    if n <= 5:
        return BasinHoppingConfig(
            n_iterations=200,
            local_iterations=20000,
            n_restarts=10,
            temperature=0.5,
        )
    elif n <= 12:
        return BasinHoppingConfig(
            n_iterations=150,
            local_iterations=15000,
            n_restarts=7,
        )
    elif n <= 25:
        return BasinHoppingConfig(
            n_iterations=100,
            local_iterations=10000,
            n_restarts=5,
        )
    else:
        return BasinHoppingConfig(
            n_iterations=50,
            local_iterations=5000,
            n_restarts=3,
        )


# =============================================================================
# STEP TAKING STRATEGIES
# =============================================================================

class AdaptiveStepTaker:
    """
    Intelligent step-taking for basin hopping.
    
    Adapts step size based on acceptance rate and history.
    Avoids recently visited regions (tabu).
    """
    
    def __init__(self, config: BasinHoppingConfig, n_trees: int):
        self.config = config
        self.n_trees = n_trees
        self.dim = n_trees * 3
        
        # Adaptive step sizes
        self.step_position = config.step_size_position
        self.step_angle = np.radians(config.step_size_angle)
        
        # History tracking
        self.history: List[np.ndarray] = []
        self.tabu_list: List[np.ndarray] = []
        
        # Acceptance tracking
        self.accept_count = 0
        self.total_count = 0
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Generate a new point by taking a random step.
        
        Strategies:
        1. Random perturbation of all variables
        2. Focus on specific trees
        3. Rotation-only moves
        4. Translation-only moves
        """
        x_new = x.copy()
        
        # Choose move strategy
        strategy = np.random.choice([
            'perturb_all',
            'perturb_subset',
            'rotate_all',
            'translate_all',
            'swap_trees',
            'shuffle_angles'
        ], p=[0.3, 0.25, 0.15, 0.15, 0.1, 0.05])
        
        if strategy == 'perturb_all':
            # Perturb all positions and angles
            for i in range(self.n_trees):
                x_new[i*3] += np.random.normal(0, self.step_position)
                x_new[i*3 + 1] += np.random.normal(0, self.step_position)
                x_new[i*3 + 2] += np.degrees(np.random.normal(0, self.step_angle))
        
        elif strategy == 'perturb_subset':
            # Perturb random subset of trees
            n_perturb = max(1, np.random.randint(1, max(2, self.n_trees // 2)))
            trees_to_perturb = np.random.choice(self.n_trees, n_perturb, replace=False)
            
            for i in trees_to_perturb:
                x_new[i*3] += np.random.normal(0, self.step_position * 2)
                x_new[i*3 + 1] += np.random.normal(0, self.step_position * 2)
                x_new[i*3 + 2] += np.degrees(np.random.normal(0, self.step_angle * 2))
        
        elif strategy == 'rotate_all':
            # Only rotate trees
            for i in range(self.n_trees):
                x_new[i*3 + 2] += np.degrees(np.random.normal(0, self.step_angle * 3))
        
        elif strategy == 'translate_all':
            # Only translate trees
            for i in range(self.n_trees):
                x_new[i*3] += np.random.normal(0, self.step_position * 2)
                x_new[i*3 + 1] += np.random.normal(0, self.step_position * 2)
        
        elif strategy == 'swap_trees' and self.n_trees >= 2:
            # Swap positions of two trees
            i, j = np.random.choice(self.n_trees, 2, replace=False)
            x_new[i*3], x_new[j*3] = x_new[j*3], x_new[i*3]
            x_new[i*3+1], x_new[j*3+1] = x_new[j*3+1], x_new[i*3+1]
        
        elif strategy == 'shuffle_angles':
            # Randomize all angles
            for i in range(self.n_trees):
                x_new[i*3 + 2] = np.random.uniform(0, 360)
        
        # Normalize angles
        for i in range(self.n_trees):
            x_new[i*3 + 2] = x_new[i*3 + 2] % 360
        
        # Check tabu list (avoid recently visited)
        if self.config.use_history and len(self.tabu_list) > 0:
            for _ in range(5):  # Try up to 5 times to avoid tabu
                is_tabu = False
                for tabu in self.tabu_list:
                    if np.linalg.norm(x_new - tabu) < self.config.history_distance_threshold:
                        is_tabu = True
                        break
                
                if not is_tabu:
                    break
                
                # Perturb more to escape tabu region
                x_new = x.copy()
                for i in range(self.n_trees):
                    x_new[i*3] += np.random.normal(0, self.step_position * 3)
                    x_new[i*3 + 1] += np.random.normal(0, self.step_position * 3)
                    x_new[i*3 + 2] += np.degrees(np.random.normal(0, self.step_angle * 3))
        
        return x_new
    
    def update_acceptance(self, accepted: bool):
        """Update acceptance statistics and adapt step size."""
        self.total_count += 1
        if accepted:
            self.accept_count += 1
        
        # Adapt step size periodically
        if self.config.adaptive_step and self.total_count % 10 == 0:
            accept_rate = self.accept_count / self.total_count
            
            if accept_rate > self.config.target_accept_rate + 0.1:
                # Too many accepts - make bigger jumps
                self.step_position *= self.config.step_adaptation_rate
                self.step_angle *= self.config.step_adaptation_rate
            elif accept_rate < self.config.target_accept_rate - 0.1:
                # Too few accepts - make smaller jumps
                self.step_position /= self.config.step_adaptation_rate
                self.step_angle /= self.config.step_adaptation_rate
            
            # Bounds
            self.step_position = np.clip(self.step_position, 0.01, 5.0)
            self.step_angle = np.clip(self.step_angle, np.radians(1), np.radians(180))
    
    def add_to_tabu(self, x: np.ndarray):
        """Add solution to tabu list."""
        self.tabu_list.append(x.copy())
        if len(self.tabu_list) > self.config.tabu_tenure:
            self.tabu_list.pop(0)
    
    def add_to_history(self, x: np.ndarray):
        """Add solution to history."""
        self.history.append(x.copy())


# =============================================================================
# LOCAL MINIMIZERS
# =============================================================================

class LocalMinimizer:
    """
    Local minimization within a basin.
    
    Supports multiple strategies:
    - SA (our custom Adaptive SA)
    - Nelder-Mead (simplex)
    - Powell (direction set)
    - L-BFGS-B (quasi-Newton)
    """
    
    def __init__(
        self,
        config: BasinHoppingConfig,
        n_trees: int,
        objective_fn: Callable,
        collision_checker: Callable
    ):
        self.config = config
        self.n_trees = n_trees
        self.objective_fn = objective_fn
        self.collision_checker = collision_checker
    
    def minimize(self, x0: np.ndarray) -> Tuple[np.ndarray, float]:
        """Run local minimization from starting point."""
        
        method = self.config.local_minimizer
        
        if method == 'sa':
            return self._minimize_sa(x0)
        elif method == 'nelder_mead':
            return self._minimize_scipy(x0, 'Nelder-Mead')
        elif method == 'powell':
            return self._minimize_scipy(x0, 'Powell')
        elif method == 'bfgs':
            return self._minimize_scipy(x0, 'L-BFGS-B')
        else:
            raise ValueError(f"Unknown minimizer: {method}")
    
    def _minimize_sa(self, x0: np.ndarray) -> Tuple[np.ndarray, float]:
        """Local minimization using our Adaptive SA."""
        from optimizers.adaptive_sa import AdaptiveSA, AdaptiveSAConfig
        
        # Convert flat params to solution
        solution = x0.reshape(self.n_trees, 3)
        
        # Quick SA config for local search
        sa_config = AdaptiveSAConfig(
            max_iterations=self.config.local_iterations,
            num_restarts=1,
            initial_temp=0.5,
            initial_step_translation=0.1,
            initial_step_rotation=10.0,
            use_multi_scale=False,  # Focus on fine moves
        )
        
        sa = AdaptiveSA(sa_config)
        result_solution, result_score = sa.optimize(
            n_trees=self.n_trees,
            initial_solution=solution,
            collision_checker=self.collision_checker,
            verbose=False
        )
        
        return result_solution.flatten(), result_score
    
    def _minimize_scipy(self, x0: np.ndarray, method: str) -> Tuple[np.ndarray, float]:
        """Local minimization using scipy."""
        from scipy.optimize import minimize
        
        result = minimize(
            self.objective_fn,
            x0,
            method=method,
            options={
                'maxiter': self.config.local_iterations // 10,
                'disp': False
            }
        )
        
        return result.x, result.fun


# =============================================================================
# BASIN HOPPING OPTIMIZER
# =============================================================================

class BasinHoppingOptimizer:
    """
    Basin Hopping optimizer for global optimization.
    
    The key insight: the energy landscape has many local minima (basins).
    We hop between basins and keep the best one found.
    
    This is especially effective when:
    - The landscape has many local minima
    - We have a good local optimizer
    - We can afford many function evaluations
    
    Example:
        optimizer = BasinHoppingOptimizer()
        solution, score = optimizer.optimize(n_trees=8)
    """
    
    def __init__(self, config: BasinHoppingConfig = None):
        self.config = config or BasinHoppingConfig()
        
        self.best_solution = None
        self.best_score = float('inf')
        self.history = []
        
        self.stats = {
            'iterations': 0,
            'accepts': 0,
            'rejects': 0,
            'restarts': 0,
            'basins_explored': 0,
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
        Optimize tree placement using Basin Hopping.
        
        Args:
            n_trees: Number of trees
            initial_solution: Starting solution (optional)
            collision_checker: Collision detection function
            tree_vertices: Tree vertices (optional)
            verbose: Override config verbosity
            
        Returns:
            (best_solution, best_score)
        """
        cfg = self.config
        if verbose is None:
            verbose = cfg.verbose
        
        # Get tree vertices
        if tree_vertices is None:
            from core.tree_polygon import TREE_VERTICES
            tree_vertices = TREE_VERTICES
        
        # Create collision checker
        if collision_checker is None:
            collision_checker = self._create_collision_checker()
        
        # Create objective function
        objective = self._create_objective(n_trees, tree_vertices, collision_checker)
        
        # Initialize
        if initial_solution is not None:
            x = initial_solution.flatten()
        else:
            x = self._create_initial(n_trees)
        
        start_time = time.time()
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Basin Hopping Optimization for n={n_trees}")
            print(f"  Iterations: {cfg.n_iterations}")
            print(f"  Restarts: {cfg.n_restarts}")
            print(f"  Local minimizer: {cfg.local_minimizer}")
            print(f"{'='*60}")
        
        # =====================================================================
        # MULTI-RESTART BASIN HOPPING
        # =====================================================================
        
        global_best_x = None
        global_best_score = float('inf')
        
        for restart in range(cfg.n_restarts):
            
            if verbose:
                print(f"\n  Restart {restart + 1}/{cfg.n_restarts}")
            
            # Initialize for this restart
            if restart == 0 and initial_solution is not None:
                x_current = initial_solution.flatten()
            elif global_best_x is not None and np.random.random() < cfg.restart_from_best:
                # Restart from best with perturbation
                x_current = global_best_x + np.random.randn(len(global_best_x)) * cfg.initial_step_size
            else:
                # Random restart
                x_current = self._create_initial(n_trees)
            
            # Create step taker for this restart
            step_taker = AdaptiveStepTaker(cfg, n_trees)
            
            # Create local minimizer
            local_minimizer = LocalMinimizer(
                cfg, n_trees, objective, collision_checker
            )
            
            # Initial local minimization
            x_current, f_current = local_minimizer.minimize(x_current)
            
            best_x = x_current.copy()
            best_score = f_current
            
            # -----------------------------------------------------------------
            # BASIN HOPPING LOOP
            # -----------------------------------------------------------------
            
            for iteration in range(cfg.n_iterations):
                # Take a random step
                x_new = step_taker(x_current)
                
                # Local minimization
                x_new, f_new = local_minimizer.minimize(x_new)
                
                self.stats['basins_explored'] += 1
                
                # Accept/reject decision
                accept = False
                
                if cfg.monotonic:
                    # Monotonic: only accept improvements
                    accept = f_new < f_current
                else:
                    # Metropolis criterion
                    if f_new < f_current:
                        accept = True
                    else:
                        delta = f_new - f_current
                        prob = math.exp(-delta / cfg.temperature)
                        accept = np.random.random() < prob
                
                # Update state
                if accept:
                    x_current = x_new
                    f_current = f_new
                    self.stats['accepts'] += 1
                    
                    if f_new < best_score:
                        best_score = f_new
                        best_x = x_new.copy()
                        
                        if verbose and iteration % 10 == 0:
                            print(f"    Iter {iteration}: ★ New best = {best_score:.8f}")
                else:
                    self.stats['rejects'] += 1
                
                # Update step taker
                step_taker.update_acceptance(accept)
                step_taker.add_to_tabu(x_current)
                
                self.stats['iterations'] += 1
            
            # Update global best
            if best_score < global_best_score:
                global_best_score = best_score
                global_best_x = best_x.copy()
                
                if verbose:
                    print(f"    Restart {restart + 1}: Global best = {global_best_score:.8f}")
            
            self.stats['restarts'] += 1
        
        # =====================================================================
        # FINALIZE
        # =====================================================================
        
        elapsed = time.time() - start_time
        self.stats['time_seconds'] = elapsed
        
        # Convert to solution format
        self.best_solution = global_best_x.reshape(n_trees, 3)
        self.best_solution[:, 2] = self.best_solution[:, 2] % 360  # Normalize angles
        self.best_score = global_best_score
        
        if verbose:
            accept_rate = self.stats['accepts'] / max(1, self.stats['accepts'] + self.stats['rejects'])
            print(f"\n  ✅ Basin Hopping Complete!")
            print(f"     Best Score: {self.best_score:.8f}")
            print(f"     Basins Explored: {self.stats['basins_explored']}")
            print(f"     Accept Rate: {accept_rate:.1%}")
            print(f"     Time: {elapsed:.1f}s")
        
        return self.best_solution, self.best_score
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def _create_collision_checker(self) -> Callable:
        """Create collision checking function."""
        def check(positions, angles_rad):
            from core.tree_polygon import transform_trees_batch, get_all_bounds
            from core.collision import check_any_collision
            
            all_vertices = transform_trees_batch(positions, angles_rad)
            all_bounds = get_all_bounds(all_vertices)
            return check_any_collision(all_vertices, all_bounds, use_shapely=True)
        
        return check
    
    def _create_objective(
        self,
        n_trees: int,
        tree_vertices: np.ndarray,
        collision_checker: Callable
    ) -> Callable:
        """Create objective function."""
        cfg = self.config
        
        def objective(params):
            solution = params.reshape(n_trees, 3)
            positions = solution[:, :2]
            angles_rad = np.radians(solution[:, 2] % 360)
            
            # Check collisions
            if collision_checker(positions, angles_rad):
                return 1e6  # Penalty
            
            # Compute score
            from core.tree_polygon import transform_trees_batch
            from core.bounding_box import compute_bounding_square
            
            all_vertices = transform_trees_batch(positions, angles_rad)
            s = compute_bounding_square(all_vertices)
            return (s * s) / n_trees
        
        return objective
    
    def _create_initial(self, n_trees: int) -> np.ndarray:
        """Create initial solution."""
        sqrt_n = int(np.ceil(np.sqrt(n_trees)))
        spacing = 1.3
        
        x0 = np.zeros(n_trees * 3)
        
        idx = 0
        for i in range(sqrt_n):
            for j in range(sqrt_n):
                if idx >= n_trees:
                    break
                x0[idx * 3] = (i - sqrt_n/2) * spacing + np.random.uniform(-0.1, 0.1)
                x0[idx * 3 + 1] = (j - sqrt_n/2) * spacing + np.random.uniform(-0.1, 0.1)
                x0[idx * 3 + 2] = np.random.uniform(0, 360)
                idx += 1
        
        return x0


# =============================================================================
# SCIPY WRAPPER
# =============================================================================

def basin_hopping_scipy(
    n_trees: int,
    initial_solution: Optional[np.ndarray] = None,
    niter: int = 100,
    temperature: float = 1.0,
    stepsize: float = 0.5,
    verbose: bool = True
) -> Tuple[np.ndarray, float]:
    """
    Use scipy's basin hopping with our objective.
    
    This provides an alternative implementation using scipy's
    well-tested basin hopping code.
    """
    from scipy.optimize import basinhopping, minimize
    from core.tree_polygon import TREE_VERTICES, transform_trees_batch, get_all_bounds
    from core.bounding_box import compute_bounding_square
    from core.collision import check_any_collision
    
    tree_vertices = TREE_VERTICES
    
    def objective(params):
        solution = params.reshape(n_trees, 3)
        positions = solution[:, :2]
        angles_rad = np.radians(solution[:, 2] % 360)
        
        all_vertices = transform_trees_batch(positions, angles_rad)
        all_bounds = get_all_bounds(all_vertices)
        
        if check_any_collision(all_vertices, all_bounds, use_shapely=True):
            return 1e6
        
        s = compute_bounding_square(all_vertices)
        return (s * s) / n_trees
    
    # Initial guess
    if initial_solution is not None:
        x0 = initial_solution.flatten()
    else:
        sqrt_n = int(np.ceil(np.sqrt(n_trees)))
        x0 = np.zeros(n_trees * 3)
        idx = 0
        for i in range(sqrt_n):
            for j in range(sqrt_n):
                if idx >= n_trees:
                    break
                x0[idx * 3] = (i - sqrt_n/2) * 1.3
                x0[idx * 3 + 1] = (j - sqrt_n/2) * 1.3
                x0[idx * 3 + 2] = np.random.uniform(0, 360)
                idx += 1
    
    # Callback for progress
    def callback(x, f, accept):
        if verbose and accept:
            print(f"  Basin hop accepted: score = {f:.8f}")
    
    # Run scipy basin hopping
    result = basinhopping(
        objective,
        x0,
        niter=niter,
        T=temperature,
        stepsize=stepsize,
        minimizer_kwargs={
            'method': 'Nelder-Mead',
            'options': {'maxiter': 1000}
        },
        callback=callback if verbose else None,
        disp=verbose
    )
    
    # Format result
    solution = result.x.reshape(n_trees, 3)
    solution[:, 2] = solution[:, 2] % 360
    
    return solution, result.fun


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def optimize_with_basin_hopping(
    n: int,
    verbose: bool = True
) -> Tuple[np.ndarray, float]:
    """
    Optimize n-tree configuration using Basin Hopping.
    
    Args:
        n: Number of trees
        verbose: Print progress
        
    Returns:
        (solution, score)
    """
    config = get_basin_hopping_config_for_n(n)
    config.verbose = verbose
    
    optimizer = BasinHoppingOptimizer(config)
    return optimizer.optimize(n_trees=n, verbose=verbose)


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    print("Testing Basin Hopping Optimizer")
    print("=" * 60)
    
    # Warm up
    print("\nWarming up JIT compilation...")
    from core.tree_polygon import warmup as warmup_tree
    warmup_tree()
    
    # Test with n=4
    test_n = 4
    print(f"\nTesting with n={test_n}...")
    
    config = BasinHoppingConfig(
        n_iterations=20,
        n_restarts=2,
        local_iterations=5000,
        local_minimizer='sa',
        verbose=True
    )
    
    optimizer = BasinHoppingOptimizer(config)
    solution, score = optimizer.optimize(n_trees=test_n, verbose=True)
    
    print(f"\n✅ Test complete!")
    print(f"   Final score: {score:.8f}")
    print(f"   Solution shape: {solution.shape}")
    
    # Test scipy wrapper
    print(f"\n{'='*60}")
    print("Testing scipy wrapper...")
    sol_scipy, score_scipy = basin_hopping_scipy(
        n_trees=3,
        niter=10,
        verbose=True
    )
    print(f"Scipy result: score = {score_scipy:.8f}")
