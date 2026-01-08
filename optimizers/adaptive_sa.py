"""
Adaptive Simulated Annealing (ASA) - The Core Optimization Engine

This is the HEART of our competition strategy. For n ≤ 40, this optimizer
is our primary weapon. Every line here is optimized for winning.

Key Innovations:
1. Modified Lam Temperature Schedule - Self-tuning to target 44% acceptance
2. Adaptive Step Sizes - Learns optimal move sizes during optimization
3. Multi-Scale Moves - Large moves for exploration, fine moves for exploitation
4. Smart Restarts - Escape deep local minima by restarting from best
5. Move Selection Strategy - Intelligent tree selection based on contribution
6. Momentum-based moves - Favor moves that have been working
7. Plateau Detection - Detect and escape flat regions

Based on research from:
- Modified Lam Annealing (Cicirello et al.)
- Adaptive Cooling SA (physics-inspired)
- Competition winning strategies from Santa 2024
"""

import numpy as np
from numba import njit, prange
from dataclasses import dataclass, field
from typing import Tuple, Optional, Callable, List, Dict
import time
import math


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class AdaptiveSAConfig:
    """
    Complete configuration for Adaptive Simulated Annealing.
    
    These parameters have been tuned based on research and experimentation.
    Different n values may benefit from different parameters - consider
    creating presets for different ranges.
    """
    
    # === Iteration Control ===
    max_iterations: int = 2_000_000
    min_iterations: int = 100_000
    convergence_threshold: float = 1e-8  # Stop if improvement < this for long time
    convergence_window: int = 50000      # Check convergence over this many iters
    
    # === Modified Lam Temperature Schedule ===
    initial_temp: float = 10.0
    min_temp: float = 1e-10
    target_accept_rate: float = 0.44     # Optimal acceptance rate (Lam)
    lam_rate: float = 0.995              # Temperature adaptation speed
    temp_check_interval: int = 100       # Check acceptance every N moves
    
    # === Step Size Parameters ===
    initial_step_translation: float = 0.3
    initial_step_rotation: float = 30.0   # degrees
    min_step_translation: float = 1e-5
    min_step_rotation: float = 0.001
    max_step_translation: float = 2.0
    max_step_rotation: float = 180.0
    step_adaptation_rate: float = 1.05   # How fast to adapt steps
    
    # === Multi-Scale Moves ===
    use_multi_scale: bool = True
    # Probabilities for each scale (must sum to 1)
    large_scale_prob: float = 0.05       # Big jumps - exploration
    medium_scale_prob: float = 0.15      # Medium moves
    small_scale_prob: float = 0.80       # Fine-tuning - exploitation
    # Scale multipliers
    large_scale_mult: float = 10.0
    medium_scale_mult: float = 3.0
    small_scale_mult: float = 1.0
    
    # === Move Type Probabilities ===
    translate_only_prob: float = 0.4     # Just move position
    rotate_only_prob: float = 0.2        # Just rotate
    combined_prob: float = 0.35          # Both translate and rotate  
    swap_prob: float = 0.05              # Swap two trees' positions
    
    # === Restart Strategy ===
    num_restarts: int = 10
    restart_threshold: int = 100000      # Restart after N iters without improvement
    restart_temp_factor: float = 0.3     # Reduce temp after restart
    restart_step_factor: float = 0.5     # Reduce step after restart
    
    # === Tree Selection Strategy ===
    use_smart_selection: bool = True
    boundary_tree_bias: float = 2.0      # Prefer trees on boundary
    recent_improve_bonus: float = 1.5    # Bonus for trees that improved recently
    
    # === Plateau Detection ===
    plateau_detection: bool = True
    plateau_threshold: float = 1e-7      # Min improvement to not be plateau
    plateau_window: int = 10000          # Window to detect plateau
    plateau_escape_mult: float = 5.0     # Increase temp/step to escape


# Strategy Presets for different n ranges
def get_config_for_n(n: int) -> AdaptiveSAConfig:
    """Get optimized configuration based on n."""
    
    if n <= 5:
        # Very small n - be extremely thorough
        return AdaptiveSAConfig(
            max_iterations=5_000_000,
            num_restarts=20,
            restart_threshold=200000,
            initial_temp=5.0,
            target_accept_rate=0.50,  # Higher acceptance for exploration
        )
    elif n <= 12:
        # Small n - still very thorough
        return AdaptiveSAConfig(
            max_iterations=3_000_000,
            num_restarts=15,
            restart_threshold=150000,
        )
    elif n <= 25:
        # Medium-small n
        return AdaptiveSAConfig(
            max_iterations=2_000_000,
            num_restarts=10,
        )
    elif n <= 40:
        # Medium n
        return AdaptiveSAConfig(
            max_iterations=1_500_000,
            num_restarts=8,
        )
    else:
        # Large n - will be refined by lattice, so less SA work
        return AdaptiveSAConfig(
            max_iterations=500_000,
            num_restarts=5,
            restart_threshold=50000,
        )


# =============================================================================
# NUMBA-OPTIMIZED CORE FUNCTIONS
# =============================================================================

@njit(cache=True, fastmath=True)
def compute_bounding_square_fast(
    positions: np.ndarray,
    angles: np.ndarray,
    tree_vertices: np.ndarray
) -> float:
    """
    Ultra-fast bounding square computation without creating intermediate arrays.
    """
    n = len(positions)
    num_verts = len(tree_vertices)
    
    min_x = 1e10
    max_x = -1e10
    min_y = 1e10
    max_y = -1e10
    
    for i in range(n):
        cx = positions[i, 0]
        cy = positions[i, 1]
        cos_a = math.cos(angles[i])
        sin_a = math.sin(angles[i])
        
        for j in range(num_verts):
            vx = tree_vertices[j, 0]
            vy = tree_vertices[j, 1]
            
            x = vx * cos_a - vy * sin_a + cx
            y = vx * sin_a + vy * cos_a + cy
            
            if x < min_x:
                min_x = x
            if x > max_x:
                max_x = x
            if y < min_y:
                min_y = y
            if y > max_y:
                max_y = y
    
    return max(max_x - min_x, max_y - min_y)


@njit(cache=True, fastmath=True)
def compute_score_fast(
    positions: np.ndarray,
    angles: np.ndarray,
    tree_vertices: np.ndarray,
    n: int
) -> float:
    """Compute s²/n score."""
    s = compute_bounding_square_fast(positions, angles, tree_vertices)
    return (s * s) / n


@njit(cache=True, fastmath=True)
def find_boundary_trees(
    positions: np.ndarray,
    angles: np.ndarray,
    tree_vertices: np.ndarray
) -> np.ndarray:
    """
    Find which trees contribute to the bounding box.
    These are the most important trees to move.
    Returns array of weights (higher = more on boundary).
    """
    n = len(positions)
    num_verts = len(tree_vertices)
    
    # Find global bounds
    global_min_x = 1e10
    global_max_x = -1e10
    global_min_y = 1e10
    global_max_y = -1e10
    
    for i in range(n):
        cx = positions[i, 0]
        cy = positions[i, 1]
        cos_a = math.cos(angles[i])
        sin_a = math.sin(angles[i])
        
        for j in range(num_verts):
            vx = tree_vertices[j, 0]
            vy = tree_vertices[j, 1]
            
            x = vx * cos_a - vy * sin_a + cx
            y = vx * sin_a + vy * cos_a + cy
            
            if x < global_min_x:
                global_min_x = x
            if x > global_max_x:
                global_max_x = x
            if y < global_min_y:
                global_min_y = y
            if y > global_max_y:
                global_max_y = y
    
    # Compute weight for each tree based on proximity to boundary
    weights = np.zeros(n, dtype=np.float64)
    tolerance = 0.01  # How close to boundary counts
    
    for i in range(n):
        cx = positions[i, 0]
        cy = positions[i, 1]
        cos_a = math.cos(angles[i])
        sin_a = math.sin(angles[i])
        
        for j in range(num_verts):
            vx = tree_vertices[j, 0]
            vy = tree_vertices[j, 1]
            
            x = vx * cos_a - vy * sin_a + cx
            y = vx * sin_a + vy * cos_a + cy
            
            # Check if this vertex is on any boundary
            if abs(x - global_min_x) < tolerance or abs(x - global_max_x) < tolerance:
                weights[i] += 1.0
            if abs(y - global_min_y) < tolerance or abs(y - global_max_y) < tolerance:
                weights[i] += 1.0
    
    # Normalize
    total = weights.sum()
    if total > 0:
        weights = weights / total
    else:
        weights = np.ones(n, dtype=np.float64) / n
    
    return weights


@njit(cache=True, fastmath=True)
def select_tree_weighted(weights: np.ndarray, rand_val: float) -> int:
    """Select a tree index based on weights."""
    cumsum = 0.0
    for i in range(len(weights)):
        cumsum += weights[i]
        if rand_val < cumsum:
            return i
    return len(weights) - 1


# =============================================================================
# ADAPTIVE SIMULATED ANNEALING OPTIMIZER
# =============================================================================

class AdaptiveSA:
    """
    Adaptive Simulated Annealing optimizer with all the bells and whistles.
    
    This is designed to WIN. It incorporates:
    - Modified Lam temperature schedule (self-tuning)
    - Adaptive step sizes
    - Multi-scale moves
    - Smart tree selection
    - Plateau detection and escape
    - Strategic restarts
    
    Example:
        optimizer = AdaptiveSA()
        solution, score = optimizer.optimize(n_trees=10)
    """
    
    def __init__(self, config: AdaptiveSAConfig = None):
        self.config = config or AdaptiveSAConfig()
        
        # State tracking
        self.best_solution = None
        self.best_score = float('inf')
        self.history = []
        self.score_history = []
        
        # Statistics
        self.stats = {
            'total_iterations': 0,
            'accepted_moves': 0,
            'rejected_moves': 0,
            'restarts': 0,
            'plateau_escapes': 0,
            'final_temp': 0,
            'final_step_trans': 0,
            'final_step_rot': 0,
        }
    
    def optimize(
        self,
        n_trees: int,
        initial_solution: Optional[np.ndarray] = None,
        collision_checker: Optional[Callable] = None,
        tree_vertices: Optional[np.ndarray] = None,
        verbose: bool = True,
        progress_callback: Optional[Callable] = None
    ) -> Tuple[np.ndarray, float]:
        """
        Optimize placement of n trees.
        
        Args:
            n_trees: Number of trees to pack
            initial_solution: Starting solution (n, 3) or None for random
            collision_checker: Function(positions, angles) -> bool for collision
            tree_vertices: Custom tree vertices or None for default
            verbose: Print progress
            progress_callback: Called with (iteration, score, best_score)
            
        Returns:
            (best_solution, best_score)
        """
        cfg = self.config
        
        # Get tree vertices
        if tree_vertices is None:
            from core.tree_polygon import TREE_VERTICES
            tree_vertices = TREE_VERTICES
        
        # Initialize solution
        if initial_solution is not None:
            positions = initial_solution[:, :2].copy()
            angles = np.radians(initial_solution[:, 2]).copy()
        else:
            positions, angles = self._create_initial_solution(n_trees)
        
        # Create collision checker if not provided
        if collision_checker is None:
            collision_checker = self._create_collision_checker()
        
        # Resolve any initial collisions
        if collision_checker(positions, angles):
            positions, angles = self._resolve_collisions(n_trees, positions, angles, collision_checker)
        
        # Initialize score tracking
        current_score = compute_score_fast(positions, angles, tree_vertices, n_trees)
        best_score = current_score
        best_positions = positions.copy()
        best_angles = angles.copy()
        
        # Initialize adaptive parameters
        temp = cfg.initial_temp
        step_trans = cfg.initial_step_translation
        step_rot = np.radians(cfg.initial_step_rotation)
        
        # Modified Lam tracking
        accept_count = 0
        move_count = 0
        
        # Convergence tracking
        last_improvement_iter = 0
        scores_window = []
        
        # Smart selection weights
        if cfg.use_smart_selection:
            tree_weights = find_boundary_trees(positions, angles, tree_vertices)
            improve_history = np.zeros(n_trees)  # Track which trees improved
        else:
            tree_weights = np.ones(n_trees) / n_trees
            improve_history = None
        
        # Restart tracking
        restart_count = 0
        
        start_time = time.time()
        
        # =====================================================================
        # MAIN OPTIMIZATION LOOP
        # =====================================================================
        
        for iteration in range(cfg.max_iterations):
            
            # -----------------------------------------------------------------
            # TREE SELECTION
            # -----------------------------------------------------------------
            if cfg.use_smart_selection and iteration % 1000 == 0:
                # Update boundary weights periodically
                tree_weights = find_boundary_trees(positions, angles, tree_vertices)
                
                # Add bonus for trees that improved recently
                if improve_history is not None:
                    tree_weights = tree_weights + improve_history * cfg.recent_improve_bonus
                    tree_weights = tree_weights / tree_weights.sum()
                    # Decay improve history
                    improve_history *= 0.99
            
            # Select tree to move
            if cfg.use_smart_selection:
                tree_idx = select_tree_weighted(tree_weights, np.random.random())
            else:
                tree_idx = np.random.randint(n_trees)
            
            # -----------------------------------------------------------------
            # MOVE GENERATION
            # -----------------------------------------------------------------
            
            # Determine move scale
            if cfg.use_multi_scale:
                r = np.random.random()
                if r < cfg.large_scale_prob:
                    scale = cfg.large_scale_mult
                elif r < cfg.large_scale_prob + cfg.medium_scale_prob:
                    scale = cfg.medium_scale_mult
                else:
                    scale = cfg.small_scale_mult
            else:
                scale = 1.0
            
            # Determine move type
            move_type = self._select_move_type()
            
            # Save old state
            old_pos = positions[tree_idx].copy()
            old_angle = angles[tree_idx]
            
            # Apply move
            if move_type == 'translate' or move_type == 'combined':
                dx = np.random.normal(0, step_trans * scale)
                dy = np.random.normal(0, step_trans * scale)
                positions[tree_idx, 0] += dx
                positions[tree_idx, 1] += dy
            
            if move_type == 'rotate' or move_type == 'combined':
                dtheta = np.random.normal(0, step_rot * scale)
                angles[tree_idx] += dtheta
                # Keep angle in [0, 2π]
                angles[tree_idx] = angles[tree_idx] % (2 * np.pi)
            
            if move_type == 'swap' and n_trees >= 2:
                # Swap with another tree
                other_idx = np.random.randint(n_trees)
                while other_idx == tree_idx:
                    other_idx = np.random.randint(n_trees)
                
                positions[tree_idx], positions[other_idx] = \
                    positions[other_idx].copy(), positions[tree_idx].copy()
            
            # -----------------------------------------------------------------
            # COLLISION CHECK
            # -----------------------------------------------------------------
            has_collision = collision_checker(positions, angles)
            
            # -----------------------------------------------------------------
            # SCORE COMPUTATION
            # -----------------------------------------------------------------
            if not has_collision:
                new_score = compute_score_fast(positions, angles, tree_vertices, n_trees)
            else:
                new_score = float('inf')  # Invalid
            
            # -----------------------------------------------------------------
            # ACCEPT/REJECT DECISION (Metropolis criterion)
            # -----------------------------------------------------------------
            accept = False
            
            if has_collision:
                accept = False
            elif new_score < current_score:
                accept = True
            else:
                # Metropolis: accept worse with probability exp(-ΔE/T)
                delta = new_score - current_score
                if temp > 1e-15:
                    prob = math.exp(-delta / temp)
                    accept = np.random.random() < prob
            
            # -----------------------------------------------------------------
            # UPDATE STATE
            # -----------------------------------------------------------------
            if accept:
                current_score = new_score
                accept_count += 1
                
                # Best solution tracking
                if new_score < best_score:
                    best_score = new_score
                    best_positions = positions.copy()
                    best_angles = angles.copy()
                    last_improvement_iter = iteration
                    
                    # Update improve history
                    if improve_history is not None:
                        improve_history[tree_idx] += 1.0
            else:
                # Revert move
                positions[tree_idx] = old_pos
                angles[tree_idx] = old_angle
                
                if move_type == 'swap' and n_trees >= 2:
                    positions[tree_idx], positions[other_idx] = \
                        positions[other_idx].copy(), positions[tree_idx].copy()
            
            move_count += 1
            
            # -----------------------------------------------------------------
            # MODIFIED LAM TEMPERATURE ADAPTATION
            # -----------------------------------------------------------------
            if move_count % cfg.temp_check_interval == 0:
                actual_rate = accept_count / cfg.temp_check_interval
                
                # Tune temperature to approach target acceptance rate
                if actual_rate > cfg.target_accept_rate:
                    temp *= cfg.lam_rate  # Too many accepts -> cool down
                else:
                    temp /= cfg.lam_rate  # Too few accepts -> heat up
                
                # Bound temperature
                temp = max(temp, cfg.min_temp)
                temp = min(temp, cfg.initial_temp * 100)
                
                # Adaptive step sizes
                if actual_rate > 0.5:
                    # Too many accepts - make bigger moves
                    step_trans = min(step_trans * cfg.step_adaptation_rate, 
                                   cfg.max_step_translation)
                    step_rot = min(step_rot * cfg.step_adaptation_rate,
                                 np.radians(cfg.max_step_rotation))
                elif actual_rate < 0.2:
                    # Too few accepts - make smaller moves
                    step_trans = max(step_trans / cfg.step_adaptation_rate,
                                   cfg.min_step_translation)
                    step_rot = max(step_rot / cfg.step_adaptation_rate,
                                 np.radians(cfg.min_step_rotation))
                
                accept_count = 0
            
            # -----------------------------------------------------------------
            # PLATEAU DETECTION & ESCAPE
            # -----------------------------------------------------------------
            if cfg.plateau_detection:
                scores_window.append(current_score)
                if len(scores_window) > cfg.plateau_window:
                    scores_window.pop(0)
                    
                    # Check if we're on a plateau
                    if len(scores_window) == cfg.plateau_window:
                        improvement = scores_window[0] - scores_window[-1]
                        if abs(improvement) < cfg.plateau_threshold:
                            # On plateau - perturb to escape
                            temp *= cfg.plateau_escape_mult
                            step_trans = min(step_trans * 2, cfg.max_step_translation)
                            step_rot = min(step_rot * 2, np.radians(cfg.max_step_rotation))
                            self.stats['plateau_escapes'] += 1
                            scores_window.clear()
            
            # -----------------------------------------------------------------
            # RESTART CHECK
            # -----------------------------------------------------------------
            if (iteration - last_improvement_iter > cfg.restart_threshold and 
                restart_count < cfg.num_restarts):
                
                if verbose:
                    elapsed = time.time() - start_time
                    print(f"  Restart {restart_count + 1}/{cfg.num_restarts} at iter {iteration:,} "
                          f"| best={best_score:.8f} | time={elapsed:.1f}s")
                
                # Restart from best solution
                positions = best_positions.copy()
                angles = best_angles.copy()
                current_score = best_score
                
                # Reduce temperature and step sizes
                temp = max(cfg.initial_temp * (cfg.restart_temp_factor ** (restart_count + 1)),
                          cfg.min_temp * 1000)
                step_trans = max(cfg.initial_step_translation * (cfg.restart_step_factor ** (restart_count + 1)),
                               cfg.min_step_translation)
                step_rot = max(np.radians(cfg.initial_step_rotation) * (cfg.restart_step_factor ** (restart_count + 1)),
                             np.radians(cfg.min_step_rotation))
                
                last_improvement_iter = iteration
                restart_count += 1
                self.stats['restarts'] = restart_count
            
            # -----------------------------------------------------------------
            # PROGRESS REPORTING
            # -----------------------------------------------------------------
            if verbose and iteration % 100000 == 0 and iteration > 0:
                elapsed = time.time() - start_time
                rate = iteration / elapsed
                print(f"  Iter {iteration:>10,} | best={best_score:.8f} | curr={current_score:.8f} | "
                      f"T={temp:.2e} | step={step_trans:.5f} | {rate:.0f} iter/s")
                
                self.score_history.append(best_score)
            
            if progress_callback:
                progress_callback(iteration, current_score, best_score)
        
        # =====================================================================
        # FINALIZE
        # =====================================================================
        
        elapsed = time.time() - start_time
        
        # Build output solution
        self.best_solution = np.column_stack([
            best_positions,
            np.degrees(best_angles)
        ])
        self.best_score = best_score
        
        # Update stats
        self.stats['total_iterations'] = cfg.max_iterations
        self.stats['final_temp'] = temp
        self.stats['final_step_trans'] = step_trans
        self.stats['final_step_rot'] = np.degrees(step_rot)
        
        if verbose:
            print(f"\n  ✅ Optimization Complete!")
            print(f"     Best Score: {best_score:.8f}")
            print(f"     Time: {elapsed:.1f}s ({cfg.max_iterations/elapsed:.0f} iter/s)")
            print(f"     Restarts: {restart_count}")
            print(f"     Plateau Escapes: {self.stats['plateau_escapes']}")
        
        return self.best_solution, self.best_score
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def _create_initial_solution(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create a collision-free initial placement.
        Uses grid pattern with some randomization.
        """
        sqrt_n = int(np.ceil(np.sqrt(n)))
        spacing = 1.3  # Conservative spacing for no collisions
        
        positions = np.zeros((n, 2), dtype=np.float64)
        angles = np.zeros(n, dtype=np.float64)
        
        idx = 0
        for i in range(sqrt_n):
            for j in range(sqrt_n):
                if idx >= n:
                    break
                # Add small random offset
                positions[idx, 0] = i * spacing + np.random.uniform(-0.1, 0.1)
                positions[idx, 1] = j * spacing + np.random.uniform(-0.1, 0.1)
                # Random rotation
                angles[idx] = np.random.uniform(0, 2 * np.pi)
                idx += 1
        
        # Center around origin
        positions[:, 0] -= positions[:, 0].mean()
        positions[:, 1] -= positions[:, 1].mean()
        
        return positions, angles
    
    def _resolve_collisions(
        self,
        n: int,
        positions: np.ndarray,
        angles: np.ndarray,
        collision_checker: Callable
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Spread trees until no collisions."""
        spread_factor = 1.0
        
        while collision_checker(positions, angles) and spread_factor < 10:
            spread_factor += 0.2
            sqrt_n = int(np.ceil(np.sqrt(n)))
            spacing = 1.3 * spread_factor
            
            idx = 0
            for i in range(sqrt_n):
                for j in range(sqrt_n):
                    if idx >= n:
                        break
                    positions[idx, 0] = (i - sqrt_n/2) * spacing
                    positions[idx, 1] = (j - sqrt_n/2) * spacing
                    idx += 1
        
        return positions, angles
    
    def _select_move_type(self) -> str:
        """Select type of move to apply."""
        cfg = self.config
        r = np.random.random()
        
        if r < cfg.translate_only_prob:
            return 'translate'
        elif r < cfg.translate_only_prob + cfg.rotate_only_prob:
            return 'rotate'
        elif r < cfg.translate_only_prob + cfg.rotate_only_prob + cfg.combined_prob:
            return 'combined'
        else:
            return 'swap'
    
    def _create_collision_checker(self) -> Callable:
        """Create a collision checking function."""
        def check(positions, angles):
            from core.tree_polygon import transform_trees_batch, get_all_bounds
            from core.collision import check_any_collision
            
            all_vertices = transform_trees_batch(positions, angles)
            all_bounds = get_all_bounds(all_vertices)
            return check_any_collision(all_vertices, all_bounds, use_shapely=True)
        
        return check


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def optimize_configuration(
    n: int,
    verbose: bool = True
) -> Tuple[np.ndarray, float]:
    """
    Optimize a single n-tree configuration using best settings.
    
    Args:
        n: Number of trees
        verbose: Print progress
        
    Returns:
        (solution, score)
    """
    config = get_config_for_n(n)
    optimizer = AdaptiveSA(config)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Optimizing n={n} trees")
        print(f"  Max iterations: {config.max_iterations:,}")
        print(f"  Restarts: {config.num_restarts}")
        print(f"{'='*60}")
    
    return optimizer.optimize(n_trees=n, verbose=verbose)


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    print("Testing Adaptive SA Optimizer")
    print("=" * 60)
    
    # Warm up JIT
    print("\nWarming up JIT compilation...")
    from core.tree_polygon import TREE_VERTICES, warmup as warmup_tree
    warmup_tree()
    
    # Quick test
    from core.collision import warmup as warmup_collision
    try:
        warmup_collision()
    except:
        pass
    
    # Test with small n
    test_n = 5
    print(f"\nRunning quick test with n={test_n}...")
    
    config = AdaptiveSAConfig(
        max_iterations=50000,
        num_restarts=2,
        restart_threshold=10000
    )
    
    optimizer = AdaptiveSA(config)
    solution, score = optimizer.optimize(n_trees=test_n, verbose=True)
    
    print(f"\n✅ Test complete!")
    print(f"   Final score: {score:.8f}")
    print(f"   Solution shape: {solution.shape}")
