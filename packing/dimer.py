"""
Dimer Optimization - The SECRET WEAPON for Large n

A "dimer" is a two-tree unit where one tree is typically rotated 180°
to interlock with the other. This creates a compact unit that TILES
EFFICIENTLY for packing many trees.

Why dimers are CRITICAL:
1. Two trees pointing in opposite directions can nest together
2. The dimer unit has a more rectangular bounding box
3. Rectangular units tile much better than individual triangular trees
4. For n > 40, dimer-based packing DOMINATES individual optimization

This module does:
1. Find the OPTIMAL relative position of tree 2 to tree 1
2. Find the OPTIMAL relative rotation
3. Find the OPTIMAL grid spacing for tiling dimers
4. Handle odd n by placing one "orphan" tree optimally

The quality of our dimer directly impacts 50%+ of our final score!

Research basis:
- Competition discussion forums identifying dimer patterns
- Tessellation theory for irregular polygons
- Evolutionary optimization for dimer parameters
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict
from scipy.optimize import differential_evolution, minimize
import time


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class DimerConfig:
    """
    Configuration defining a two-tree dimer unit.
    
    Tree 1: At origin (0, 0) with angle = 0
    Tree 2: At (dx, dy) with angle = angle2
    
    The dimer itself can be globally rotated by global_angle.
    """
    
    # Position of second tree relative to first
    dx: float = 0.0
    dy: float = -0.55  # Typically below (nest into each other)
    
    # Rotation of second tree (degrees)
    angle2: float = 180.0  # Typically inverted
    
    # Global rotation of entire dimer (degrees)
    global_angle: float = 0.0
    
    # Grid spacing for tiling
    spacing_x: float = 0.75
    spacing_y: float = 1.0
    
    # Offset for alternating rows (brick-like pattern)
    row_offset: float = 0.0
    
    # Quality metrics (computed)
    bounding_area: float = 0.0
    packing_efficiency: float = 0.0

    def to_dict(self) -> Dict:
        return {
            'dx': self.dx,
            'dy': self.dy,
            'angle2': self.angle2,
            'global_angle': self.global_angle,
            'spacing_x': self.spacing_x,
            'spacing_y': self.spacing_y,
            'row_offset': self.row_offset,
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'DimerConfig':
        return cls(**d)


@dataclass
class DimerOptimizationConfig:
    """Configuration for dimer optimization."""
    
    # Differential evolution parameters
    de_maxiter: int = 1000
    de_popsize: int = 15
    de_mutation: Tuple[float, float] = (0.5, 1.0)
    de_recombination: float = 0.7
    de_workers: int = 1  # Must be 1 - local functions can't be pickled for multiprocessing
    
    # Search bounds
    dx_bounds: Tuple[float, float] = (-0.5, 0.5)
    dy_bounds: Tuple[float, float] = (-1.0, 0.0)
    angle2_bounds: Tuple[float, float] = (150.0, 210.0)
    global_angle_bounds: Tuple[float, float] = (0.0, 90.0)
    
    # Grid spacing optimization
    optimize_spacing: bool = True
    spacing_margin: float = 0.02  # Safety margin above minimum
    
    # Fine-tuning
    use_local_refinement: bool = True
    refinement_iterations: int = 100
    
    # Verbose
    verbose: bool = True


# =============================================================================
# DIMER GEOMETRY FUNCTIONS
# =============================================================================

def get_dimer_vertices(
    config: DimerConfig,
    tree_vertices: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get vertices of both trees in a dimer.
    
    Args:
        config: Dimer configuration
        tree_vertices: Base tree vertices
        
    Returns:
        (tree1_vertices, tree2_vertices) - both transformed
    """
    from core.tree_polygon import transform_tree
    
    global_rad = np.radians(config.global_angle)
    
    # Tree 1: at origin, only global rotation
    verts1 = transform_tree(0, 0, global_rad)
    
    # Tree 2: offset and rotated
    # First apply global rotation to the offset
    cos_g = np.cos(global_rad)
    sin_g = np.sin(global_rad)
    
    dx_rot = config.dx * cos_g - config.dy * sin_g
    dy_rot = config.dx * sin_g + config.dy * cos_g
    
    # Tree 2 total angle
    angle2_rad = global_rad + np.radians(config.angle2)
    
    verts2 = transform_tree(dx_rot, dy_rot, angle2_rad)
    
    return verts1, verts2


def compute_dimer_bounds(
    config: DimerConfig,
    tree_vertices: np.ndarray
) -> Tuple[float, float, float, float]:
    """
    Compute bounding box of a dimer.
    
    Returns:
        (min_x, min_y, max_x, max_y)
    """
    verts1, verts2 = get_dimer_vertices(config, tree_vertices)
    
    all_verts = np.vstack([verts1, verts2])
    
    return (
        all_verts[:, 0].min(),
        all_verts[:, 1].min(),
        all_verts[:, 0].max(),
        all_verts[:, 1].max()
    )


def compute_dimer_bounding_area(
    config: DimerConfig,
    tree_vertices: np.ndarray
) -> float:
    """Compute bounding rectangle area of a dimer."""
    min_x, min_y, max_x, max_y = compute_dimer_bounds(config, tree_vertices)
    return (max_x - min_x) * (max_y - min_y)


def check_dimer_collision(
    config: DimerConfig,
    tree_vertices: np.ndarray
) -> bool:
    """Check if the two trees in a dimer collide."""
    from core.collision import shapely_overlap
    
    verts1, verts2 = get_dimer_vertices(config, tree_vertices)
    return shapely_overlap(verts1, verts2)


def compute_minimum_spacing(
    config: DimerConfig,
    tree_vertices: np.ndarray,
    margin: float = 0.02
) -> Tuple[float, float]:
    """
    Compute minimum grid spacing to avoid collisions between adjacent dimers.
    
    Args:
        config: Dimer configuration
        tree_vertices: Base tree vertices
        margin: Safety margin
        
    Returns:
        (spacing_x, spacing_y)
    """
    min_x, min_y, max_x, max_y = compute_dimer_bounds(config, tree_vertices)
    
    spacing_x = (max_x - min_x) + margin
    spacing_y = (max_y - min_y) + margin
    
    return spacing_x, spacing_y


# =============================================================================
# DIMER OPTIMIZER
# =============================================================================

class DimerOptimizer:
    """
    Optimizer to find the BEST dimer configuration.
    
    Uses differential evolution for global optimization of:
    - Relative position (dx, dy)
    - Relative rotation (angle2)
    - Global rotation (global_angle)
    
    The objective is to minimize the bounding area while avoiding collisions.
    
    Example:
        optimizer = DimerOptimizer()
        best_config = optimizer.optimize()
        print(f"Best area: {best_config.bounding_area}")
    """
    
    def __init__(self, config: DimerOptimizationConfig = None):
        self.config = config or DimerOptimizationConfig()
        self.tree_vertices = None
        self.best_config = None
        self.history = []
    
    def optimize(
        self,
        tree_vertices: np.ndarray = None,
        verbose: bool = None
    ) -> DimerConfig:
        """
        Find the optimal dimer configuration.
        
        Args:
            tree_vertices: Tree shape vertices (uses default if None)
            verbose: Override config verbosity
            
        Returns:
            Optimal DimerConfig
        """
        cfg = self.config
        if verbose is None:
            verbose = cfg.verbose
        
        # Get tree vertices
        if tree_vertices is None:
            from core.tree_polygon import TREE_VERTICES
            tree_vertices = TREE_VERTICES
        self.tree_vertices = tree_vertices
        
        start_time = time.time()
        
        if verbose:
            print(f"\n{'='*60}")
            print("Dimer Optimization")
            print(f"  DE iterations: {cfg.de_maxiter}")
            print(f"  Population: {cfg.de_popsize}")
            print(f"{'='*60}")
        
        # =====================================================================
        # DIFFERENTIAL EVOLUTION
        # =====================================================================
        
        bounds = [
            cfg.dx_bounds,
            cfg.dy_bounds,
            cfg.angle2_bounds,
            cfg.global_angle_bounds,
        ]
        
        def objective(params):
            dx, dy, angle2, global_angle = params
            
            dimer = DimerConfig(
                dx=dx,
                dy=dy,
                angle2=angle2,
                global_angle=global_angle
            )
            
            # Check for collision between the two trees
            if check_dimer_collision(dimer, tree_vertices):
                return 1e6  # Penalty
            
            # Objective: minimize bounding area
            area = compute_dimer_bounding_area(dimer, tree_vertices)
            
            return area
        
        # Run differential evolution
        result = differential_evolution(
            objective,
            bounds,
            maxiter=cfg.de_maxiter,
            popsize=cfg.de_popsize,
            mutation=cfg.de_mutation,
            recombination=cfg.de_recombination,
            workers=cfg.de_workers,
            seed=42,
            disp=verbose,
            polish=cfg.use_local_refinement,
        )
        
        # Build best config
        dx, dy, angle2, global_angle = result.x
        
        best_dimer = DimerConfig(
            dx=dx,
            dy=dy,
            angle2=angle2,
            global_angle=global_angle
        )
        
        # =====================================================================
        # OPTIMIZE GRID SPACING
        # =====================================================================
        
        if cfg.optimize_spacing:
            spacing_x, spacing_y = compute_minimum_spacing(
                best_dimer, tree_vertices, cfg.spacing_margin
            )
            best_dimer.spacing_x = spacing_x
            best_dimer.spacing_y = spacing_y
            
            # Try brick-like offset
            best_offset, best_offset_score = self._optimize_row_offset(
                best_dimer, tree_vertices
            )
            best_dimer.row_offset = best_offset
        
        # Compute metrics
        best_dimer.bounding_area = compute_dimer_bounding_area(best_dimer, tree_vertices)
        
        # Two trees per dimer
        from core.tree_polygon import TREE_AREA
        best_dimer.packing_efficiency = (2 * TREE_AREA) / best_dimer.bounding_area
        
        elapsed = time.time() - start_time
        
        self.best_config = best_dimer
        
        if verbose:
            print(f"\n  ✅ Dimer Optimization Complete!")
            print(f"     dx={best_dimer.dx:.4f}, dy={best_dimer.dy:.4f}")
            print(f"     angle2={best_dimer.angle2:.2f}°")
            print(f"     global_angle={best_dimer.global_angle:.2f}°")
            print(f"     spacing=({best_dimer.spacing_x:.4f}, {best_dimer.spacing_y:.4f})")
            print(f"     row_offset={best_dimer.row_offset:.4f}")
            print(f"     Bounding area: {best_dimer.bounding_area:.4f}")
            print(f"     Packing efficiency: {best_dimer.packing_efficiency:.1%}")
            print(f"     Time: {elapsed:.1f}s")
        
        return best_dimer
    
    def _optimize_row_offset(
        self,
        dimer: DimerConfig,
        tree_vertices: np.ndarray
    ) -> Tuple[float, float]:
        """
        Optimize row offset for brick-like pattern.
        
        Returns:
            (best_offset, best_score)
        """
        best_offset = 0.0
        best_score = float('inf')
        
        # Try different offsets
        for offset_fraction in np.linspace(0, 0.5, 11):
            offset = offset_fraction * dimer.spacing_x
            
            # Simulate a 3x3 grid and check for collisions + density
            score = self._evaluate_grid_pattern(dimer, tree_vertices, offset)
            
            if score < best_score:
                best_score = score
                best_offset = offset
        
        return best_offset, best_score
    
    def _evaluate_grid_pattern(
        self,
        dimer: DimerConfig,
        tree_vertices: np.ndarray,
        row_offset: float
    ) -> float:
        """Evaluate a grid pattern with given row offset."""
        from core.collision import shapely_overlap
        
        # Generate a small test grid (3x3 dimers)
        all_verts = []
        
        for row in range(3):
            for col in range(3):
                x_base = col * dimer.spacing_x + (row % 2) * row_offset
                y_base = row * dimer.spacing_y
                
                # Get dimer at this position
                verts1, verts2 = get_dimer_vertices(dimer, tree_vertices)
                
                # Translate
                verts1 = verts1 + np.array([x_base, y_base])
                verts2 = verts2 + np.array([x_base, y_base])
                
                all_verts.append(verts1)
                all_verts.append(verts2)
        
        # Check for any collisions
        n_trees = len(all_verts)
        collision_count = 0
        
        for i in range(n_trees):
            for j in range(i + 1, n_trees):
                if shapely_overlap(all_verts[i], all_verts[j]):
                    collision_count += 1
        
        # Penalty for collisions
        if collision_count > 0:
            return 1e6 + collision_count
        
        # Score based on bounding box of the grid
        all_points = np.vstack(all_verts)
        width = all_points[:, 0].max() - all_points[:, 0].min()
        height = all_points[:, 1].max() - all_points[:, 1].min()
        
        return width * height


# =============================================================================
# PREDEFINED DIMERS
# =============================================================================

def get_classic_dimer() -> DimerConfig:
    """
    Classic dimer: one tree inverted directly below the other.
    
    This is a reasonable starting point before optimization.
    """
    return DimerConfig(
        dx=0.0,
        dy=-0.55,
        angle2=180.0,
        global_angle=0.0,
        spacing_x=0.75,
        spacing_y=1.1,
    )


def get_rotated_dimer() -> DimerConfig:
    """
    Rotated dimer: the pair is rotated for different tiling properties.
    """
    return DimerConfig(
        dx=0.1,
        dy=-0.5,
        angle2=180.0,
        global_angle=45.0,
        spacing_x=0.85,
        spacing_y=0.95,
    )


def get_offset_dimer() -> DimerConfig:
    """
    Offset dimer: trees are offset horizontally for interlocking.
    """
    return DimerConfig(
        dx=0.2,
        dy=-0.45,
        angle2=175.0,
        global_angle=15.0,
        spacing_x=0.8,
        spacing_y=1.0,
    )


# =============================================================================
# MULTI-DIMER OPTIMIZATION
# =============================================================================

def optimize_multiple_dimers(
    n_candidates: int = 10,
    verbose: bool = True
) -> List[DimerConfig]:
    """
    Find multiple diverse dimer configurations.
    
    Useful for testing which dimer works best for different n values.
    
    Args:
        n_candidates: Number of different dimers to find
        verbose: Print progress
        
    Returns:
        List of DimerConfig sorted by bounding area
    """
    from core.tree_polygon import TREE_VERTICES
    
    candidates = []
    
    for i in range(n_candidates):
        if verbose:
            print(f"\nOptimizing dimer {i+1}/{n_candidates}...")
        
        # Different random seeds -> different solutions
        config = DimerOptimizationConfig(
            de_maxiter=500,
            verbose=False,
        )
        
        optimizer = DimerOptimizer(config)
        
        # Slightly perturb bounds for diversity
        np.random.seed(i * 42)
        
        dimer = optimizer.optimize(TREE_VERTICES, verbose=False)
        candidates.append(dimer)
        
        if verbose:
            print(f"  Area: {dimer.bounding_area:.4f}, "
                  f"Efficiency: {dimer.packing_efficiency:.1%}")
    
    # Sort by area
    candidates.sort(key=lambda d: d.bounding_area)
    
    return candidates


# =============================================================================
# DIMER EVALUATION
# =============================================================================

def evaluate_dimer_for_n(
    dimer: DimerConfig,
    n: int,
    tree_vertices: np.ndarray = None
) -> Tuple[np.ndarray, float]:
    """
    Evaluate how well a dimer packs n trees.
    
    Args:
        dimer: Dimer configuration to use
        n: Number of trees
        tree_vertices: Tree vertices
        
    Returns:
        (solution, score)
    """
    if tree_vertices is None:
        from core.tree_polygon import TREE_VERTICES
        tree_vertices = TREE_VERTICES
    
    # Number of dimers needed
    n_dimers = (n + 1) // 2
    has_orphan = (n % 2 == 1)
    
    # Grid dimensions
    grid_cols = int(np.ceil(np.sqrt(n_dimers)))
    grid_rows = int(np.ceil(n_dimers / grid_cols))
    
    # Generate solution
    solution = []
    dimer_count = 0
    
    global_rad = np.radians(dimer.global_angle)
    cos_g = np.cos(global_rad)
    sin_g = np.sin(global_rad)
    
    for row in range(grid_rows):
        for col in range(grid_cols):
            if dimer_count >= n_dimers:
                break
            
            # Base position
            x_base = col * dimer.spacing_x + (row % 2) * dimer.row_offset
            y_base = row * dimer.spacing_y
            
            # Tree 1
            if len(solution) < n:
                solution.append([x_base, y_base, dimer.global_angle])
            
            # Tree 2
            if len(solution) < n:
                # Offset relative to tree 1
                dx_rot = dimer.dx * cos_g - dimer.dy * sin_g
                dy_rot = dimer.dx * sin_g + dimer.dy * cos_g
                
                x2 = x_base + dx_rot
                y2 = y_base + dy_rot
                angle2 = dimer.global_angle + dimer.angle2
                
                solution.append([x2, y2, angle2])
            
            dimer_count += 1
    
    solution = np.array(solution)
    
    # Center
    solution[:, 0] -= solution[:, 0].mean()
    solution[:, 1] -= solution[:, 1].mean()
    
    # Compute score
    from core.tree_polygon import transform_trees_batch
    from core.bounding_box import compute_score
    
    positions = solution[:, :2]
    angles_rad = np.radians(solution[:, 2])
    
    all_vertices = transform_trees_batch(positions, angles_rad)
    score = compute_score(all_vertices, n)
    
    return solution, score


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_dimer(dimer: DimerConfig, save_path: str = None):
    """
    Visualize a dimer configuration.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon
    from core.tree_polygon import TREE_VERTICES
    
    verts1, verts2 = get_dimer_vertices(dimer, TREE_VERTICES)
    
    fig, ax = plt.subplots(figsize=(8, 10))
    
    # Tree 1
    poly1 = MplPolygon(verts1, closed=True, facecolor='green', 
                       edgecolor='darkgreen', alpha=0.7, linewidth=2)
    ax.add_patch(poly1)
    
    # Tree 2
    poly2 = MplPolygon(verts2, closed=True, facecolor='forestgreen',
                       edgecolor='darkgreen', alpha=0.7, linewidth=2)
    ax.add_patch(poly2)
    
    # Bounding box
    min_x, min_y, max_x, max_y = compute_dimer_bounds(dimer, TREE_VERTICES)
    rect = plt.Rectangle((min_x, min_y), max_x - min_x, max_y - min_y,
                         fill=False, edgecolor='blue', linestyle='--', linewidth=2)
    ax.add_patch(rect)
    
    # Labels
    ax.plot(0, 0, 'ro', markersize=8, label='Tree 1 center')
    
    global_rad = np.radians(dimer.global_angle)
    cos_g = np.cos(global_rad)
    sin_g = np.sin(global_rad)
    dx_rot = dimer.dx * cos_g - dimer.dy * sin_g
    dy_rot = dimer.dx * sin_g + dimer.dy * cos_g
    ax.plot(dx_rot, dy_rot, 'bo', markersize=8, label='Tree 2 center')
    
    ax.set_aspect('equal')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Dimer: area={dimer.bounding_area:.4f}, eff={dimer.packing_efficiency:.1%}")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    
    plt.show()


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    print("Testing Dimer Optimizer")
    print("=" * 60)
    
    # Warm up
    print("\nWarming up...")
    from core.tree_polygon import TREE_VERTICES, warmup as warmup_tree
    warmup_tree()
    
    # Test predefined dimers
    print("\n" + "="*60)
    print("Predefined Dimers:")
    
    for name, dimer_fn in [
        ("Classic", get_classic_dimer),
        ("Rotated", get_rotated_dimer),
        ("Offset", get_offset_dimer),
    ]:
        dimer = dimer_fn()
        
        # Check for collision
        if check_dimer_collision(dimer, TREE_VERTICES):
            print(f"  {name}: COLLISION!")
            continue
        
        area = compute_dimer_bounding_area(dimer, TREE_VERTICES)
        print(f"  {name}: area={area:.4f}")
    
    # Run optimization
    print("\n" + "="*60)
    print("Running Dimer Optimization...")
    
    config = DimerOptimizationConfig(
        de_maxiter=200,
        verbose=True
    )
    
    optimizer = DimerOptimizer(config)
    best_dimer = optimizer.optimize(verbose=True)
    
    # Test packing with optimized dimer
    print("\n" + "="*60)
    print("Testing dimer for different n values:")
    
    for test_n in [10, 20, 50, 100]:
        solution, score = evaluate_dimer_for_n(best_dimer, test_n)
        print(f"  n={test_n:3d}: score={score:.6f}")
    
    print("\n✅ Test complete!")
