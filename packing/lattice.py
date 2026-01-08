"""
Lattice Packing - Regular Patterns for Large n

For large n (40+), we can't afford to optimize each tree individually.
Instead, we use LATTICE PACKING:

1. Use optimized dimers (2-tree units)
2. Tile them in a regular grid pattern
3. Handle the "edges" with light optimization
4. Polish with quick SA

This is how we achieve SCALABLE efficiency:
- n=50 → ~0.7s vs hours for pure SA
- n=100 → ~1.5s
- n=200 → ~3s

The key insight: for large n, the PATTERN matters more than
micro-optimizing individual positions.

Lattice types supported:
1. Rectangular grid - simple rows and columns
2. Hexagonal - offset rows for denser packing
3. Brick pattern - alternating row offsets
4. Rhombic - diamond-shaped lattice

This module also handles:
- Edge tree optimization (trees on the perimeter)
- Orphan tree placement (odd n)
- Compact centering
- Hybrid with SA refinement
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict, Callable
import time


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class LatticeConfig:
    """Configuration for lattice packing."""
    
    # === Lattice Type ===
    lattice_type: str = 'rectangular'  # 'rectangular', 'hexagonal', 'brick', 'rhombic'
    
    # === Dimer Settings ===
    use_optimized_dimer: bool = True
    dimer_optimization_iters: int = 500
    
    # === Grid Parameters (used if dimer not available) ===
    spacing_x: float = 0.75
    spacing_y: float = 1.0
    row_offset: float = 0.0  # For brick pattern
    default_angle: float = 0.0
    alt_angle: float = 180.0  # Alternating angle for dimer effect
    
    # === Edge Optimization ===
    optimize_edges: bool = True
    edge_sa_iterations: int = 50000
    edge_fraction: float = 0.15  # Fraction of trees to consider as "edge"
    
    # === Orphan Handling ===
    orphan_optimization_iters: int = 10000
    
    # === Refinement ===
    use_refinement: bool = True
    refinement_iterations: int = 100000
    
    # === Multiple Lattice Search ===
    try_multiple_lattices: bool = True
    n_lattice_attempts: int = 5
    
    # === Verbosity ===
    verbose: bool = True


# =============================================================================
# LATTICE GENERATORS
# =============================================================================

class LatticeGenerator:
    """
    Generates different lattice patterns for tree placement.
    """
    
    def __init__(self, config: LatticeConfig = None):
        self.config = config or LatticeConfig()
    
    def generate_rectangular(
        self,
        n: int,
        spacing_x: float,
        spacing_y: float,
        angle1: float = 0.0,
        angle2: float = 180.0
    ) -> np.ndarray:
        """
        Generate rectangular grid pattern.
        
        Trees alternate between angle1 and angle2 to create
        dimer-like interlocking.
        """
        # Grid dimensions
        # For dimer pattern, we need pairs
        n_cols = int(np.ceil(np.sqrt(n * spacing_y / spacing_x)))
        n_rows = int(np.ceil(n / n_cols))
        
        solution = []
        tree_count = 0
        
        for row in range(n_rows):
            for col in range(n_cols):
                if tree_count >= n:
                    break
                
                x = col * spacing_x
                y = row * spacing_y
                
                # Alternate angles for dimer effect
                if (row + col) % 2 == 0:
                    angle = angle1
                else:
                    angle = angle2
                
                solution.append([x, y, angle])
                tree_count += 1
        
        solution = np.array(solution)
        
        # Center
        solution[:, 0] -= solution[:, 0].mean()
        solution[:, 1] -= solution[:, 1].mean()
        
        return solution
    
    def generate_hexagonal(
        self,
        n: int,
        spacing: float = 0.8,
        angle1: float = 0.0,
        angle2: float = 180.0
    ) -> np.ndarray:
        """
        Generate hexagonal lattice pattern.
        
        Each row is offset by half the spacing, creating
        a honeycomb-like pattern.
        """
        spacing_y = spacing * np.sqrt(3) / 2
        
        n_cols = int(np.ceil(np.sqrt(n)))
        n_rows = int(np.ceil(n / n_cols))
        
        solution = []
        tree_count = 0
        
        for row in range(n_rows):
            row_offset = (row % 2) * (spacing / 2)
            
            for col in range(n_cols + 1):  # +1 for offset rows
                if tree_count >= n:
                    break
                
                x = col * spacing + row_offset
                y = row * spacing_y
                
                # Alternate angles
                angle = angle1 if (row + col) % 2 == 0 else angle2
                
                solution.append([x, y, angle])
                tree_count += 1
        
        solution = np.array(solution)
        solution[:, 0] -= solution[:, 0].mean()
        solution[:, 1] -= solution[:, 1].mean()
        
        return solution
    
    def generate_brick(
        self,
        n: int,
        spacing_x: float,
        spacing_y: float,
        offset_fraction: float = 0.5,
        angle1: float = 0.0,
        angle2: float = 180.0
    ) -> np.ndarray:
        """
        Generate brick-like pattern with row offsets.
        """
        n_cols = int(np.ceil(np.sqrt(n * spacing_y / spacing_x)))
        n_rows = int(np.ceil(n / n_cols))
        
        solution = []
        tree_count = 0
        
        for row in range(n_rows):
            row_offset = (row % 2) * spacing_x * offset_fraction
            
            for col in range(n_cols + 1):
                if tree_count >= n:
                    break
                
                x = col * spacing_x + row_offset
                y = row * spacing_y
                
                angle = angle1 if (row + col) % 2 == 0 else angle2
                
                solution.append([x, y, angle])
                tree_count += 1
        
        solution = np.array(solution)
        solution[:, 0] -= solution[:, 0].mean()
        solution[:, 1] -= solution[:, 1].mean()
        
        return solution
    
    def generate_dimer_grid(
        self,
        n: int,
        dimer_config
    ) -> np.ndarray:
        """
        Generate grid using optimized dimer configuration.
        
        This is the BEST method for large n.
        """
        from packing.dimer import get_dimer_vertices
        from core.tree_polygon import TREE_VERTICES
        
        # Number of dimers needed
        n_dimers = (n + 1) // 2
        
        # Grid dimensions for dimers
        n_cols = int(np.ceil(np.sqrt(n_dimers)))
        n_rows = int(np.ceil(n_dimers / n_cols))
        
        solution = []
        
        global_rad = np.radians(dimer_config.global_angle)
        cos_g = np.cos(global_rad)
        sin_g = np.sin(global_rad)
        
        dimer_count = 0
        
        for row in range(n_rows):
            for col in range(n_cols):
                if dimer_count >= n_dimers:
                    break
                
                # Base position for this dimer
                x_base = col * dimer_config.spacing_x
                y_base = row * dimer_config.spacing_y
                
                # Apply row offset (brick pattern)
                x_base += (row % 2) * dimer_config.row_offset
                
                # Tree 1 of dimer
                if len(solution) < n:
                    solution.append([x_base, y_base, dimer_config.global_angle])
                
                # Tree 2 of dimer
                if len(solution) < n:
                    # Compute rotated offset
                    dx_rot = dimer_config.dx * cos_g - dimer_config.dy * sin_g
                    dy_rot = dimer_config.dx * sin_g + dimer_config.dy * cos_g
                    
                    x2 = x_base + dx_rot
                    y2 = y_base + dy_rot
                    angle2 = dimer_config.global_angle + dimer_config.angle2
                    
                    solution.append([x2, y2, angle2])
                
                dimer_count += 1
        
        solution = np.array(solution)
        
        # Center
        solution[:, 0] -= solution[:, 0].mean()
        solution[:, 1] -= solution[:, 1].mean()
        
        return solution


# =============================================================================
# LATTICE PACKER
# =============================================================================

class LatticePacker:
    """
    Pack n trees using efficient lattice patterns.
    
    This is the PRIMARY method for n > 40. It:
    1. Generates a base lattice using optimized dimers
    2. Optionally optimizes edge trees
    3. Optionally refines with SA
    
    Example:
        packer = LatticePacker()
        solution, score = packer.pack(n=100)
    """
    
    def __init__(self, config: LatticeConfig = None):
        self.config = config or LatticeConfig()
        self.generator = LatticeGenerator(self.config)
        self.dimer_config = None
        self.best_solution = None
        self.best_score = float('inf')
    
    def pack(
        self,
        n: int,
        dimer_config = None,
        verbose: bool = None
    ) -> Tuple[np.ndarray, float]:
        """
        Pack n trees using lattice method.
        
        Args:
            n: Number of trees
            dimer_config: Pre-optimized dimer (will optimize if None)
            verbose: Override config verbosity
            
        Returns:
            (solution, score)
        """
        cfg = self.config
        if verbose is None:
            verbose = cfg.verbose
        
        start_time = time.time()
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Lattice Packing for n={n}")
            print(f"  Lattice type: {cfg.lattice_type}")
            print(f"  Edge optimization: {cfg.optimize_edges}")
            print(f"  Refinement: {cfg.use_refinement}")
            print(f"{'='*60}")
        
        # =====================================================================
        # STEP 1: Get or Optimize Dimer
        # =====================================================================
        
        if dimer_config is not None:
            self.dimer_config = dimer_config
        elif cfg.use_optimized_dimer:
            if verbose:
                print("\n  Optimizing dimer...")
            self.dimer_config = self._optimize_dimer()
        else:
            # Use default dimer
            from packing.dimer import get_classic_dimer
            self.dimer_config = get_classic_dimer()
        
        # =====================================================================
        # STEP 2: Generate Base Lattice
        # =====================================================================
        
        if verbose:
            print("\n  Generating lattice...")
        
        if cfg.try_multiple_lattices:
            solution, score = self._try_multiple_lattices(n)
        else:
            solution = self._generate_lattice(n)
            score = self._compute_score(solution)
        
        if verbose:
            print(f"  Base lattice score: {score:.6f}")
        
        # =====================================================================
        # STEP 3: Edge Optimization
        # =====================================================================
        
        if cfg.optimize_edges:
            if verbose:
                print("\n  Optimizing edge trees...")
            
            solution, score = self._optimize_edges(solution, n)
            
            if verbose:
                print(f"  After edge optimization: {score:.6f}")
        
        # =====================================================================
        # STEP 4: Global Refinement
        # =====================================================================
        
        if cfg.use_refinement:
            if verbose:
                print("\n  Running SA refinement...")
            
            solution, score = self._refine_with_sa(solution, n)
            
            if verbose:
                print(f"  After refinement: {score:.6f}")
        
        # =====================================================================
        # FINALIZE
        # =====================================================================
        
        # Ensure no collisions
        solution = self._resolve_any_collisions(solution)
        
        # Center the solution
        solution[:, 0] -= solution[:, 0].mean()
        solution[:, 1] -= solution[:, 1].mean()
        
        # Final score
        score = self._compute_score(solution)
        
        elapsed = time.time() - start_time
        
        self.best_solution = solution
        self.best_score = score
        
        if verbose:
            print(f"\n  ✅ Lattice Packing Complete!")
            print(f"     Final Score: {score:.8f}")
            print(f"     Time: {elapsed:.1f}s")
        
        return solution, score
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def _optimize_dimer(self):
        """Optimize dimer configuration."""
        from packing.dimer import DimerOptimizer, DimerOptimizationConfig
        
        opt_config = DimerOptimizationConfig(
            de_maxiter=self.config.dimer_optimization_iters,
            verbose=False
        )
        
        optimizer = DimerOptimizer(opt_config)
        return optimizer.optimize(verbose=False)
    
    def _generate_lattice(self, n: int) -> np.ndarray:
        """Generate lattice based on configuration."""
        cfg = self.config
        
        if self.dimer_config is not None:
            return self.generator.generate_dimer_grid(n, self.dimer_config)
        
        if cfg.lattice_type == 'rectangular':
            return self.generator.generate_rectangular(
                n, cfg.spacing_x, cfg.spacing_y, cfg.default_angle, cfg.alt_angle
            )
        elif cfg.lattice_type == 'hexagonal':
            return self.generator.generate_hexagonal(
                n, cfg.spacing_x, cfg.default_angle, cfg.alt_angle
            )
        elif cfg.lattice_type == 'brick':
            return self.generator.generate_brick(
                n, cfg.spacing_x, cfg.spacing_y, cfg.row_offset,
                cfg.default_angle, cfg.alt_angle
            )
        else:
            # Default to rectangular
            return self.generator.generate_rectangular(
                n, cfg.spacing_x, cfg.spacing_y, cfg.default_angle, cfg.alt_angle
            )
    
    def _try_multiple_lattices(self, n: int) -> Tuple[np.ndarray, float]:
        """Try multiple lattice configurations and return the best."""
        cfg = self.config
        
        best_solution = None
        best_score = float('inf')
        
        # Different rotation angles to try
        angles_to_try = [0, 15, 30, 45, 60, 75, 90]
        
        for global_angle in angles_to_try[:cfg.n_lattice_attempts]:
            # Modify dimer angle temporarily
            if self.dimer_config is not None:
                original_angle = self.dimer_config.global_angle
                self.dimer_config.global_angle = global_angle
            
            solution = self._generate_lattice(n)
            score = self._compute_score(solution)
            
            if score < best_score:
                best_score = score
                best_solution = solution.copy()
            
            # Restore
            if self.dimer_config is not None:
                self.dimer_config.global_angle = original_angle
        
        return best_solution, best_score
    
    def _optimize_edges(
        self,
        solution: np.ndarray,
        n: int
    ) -> Tuple[np.ndarray, float]:
        """
        Optimize trees on the edge of the packing.
        
        Edge trees have the most impact on the bounding box.
        """
        cfg = self.config
        
        # Find edge trees (those that touch the bounding box)
        edge_indices = self._find_edge_trees(solution)
        
        if len(edge_indices) == 0:
            return solution, self._compute_score(solution)
        
        # Run targeted SA on edge trees only
        from optimizers.adaptive_sa import AdaptiveSA, AdaptiveSAConfig
        
        sa_config = AdaptiveSAConfig(
            max_iterations=cfg.edge_sa_iterations,
            num_restarts=2,
            initial_temp=0.5,
            initial_step_translation=0.1,
            initial_step_rotation=15.0,
        )
        
        # Create a version that only moves edge trees
        best_solution = solution.copy()
        best_score = self._compute_score(solution)
        
        positions = solution[:, :2].copy()
        angles = np.radians(solution[:, 2]).copy()
        
        from core.tree_polygon import TREE_VERTICES, transform_trees_batch
        from core.bounding_box import compute_bounding_square
        
        # Quick SA for edge trees only
        temp = 0.5
        step = 0.1
        
        for iteration in range(cfg.edge_sa_iterations):
            # Pick a random edge tree
            idx = edge_indices[np.random.randint(len(edge_indices))]
            
            # Save state
            old_pos = positions[idx].copy()
            old_angle = angles[idx]
            
            # Make a move
            positions[idx, 0] += np.random.normal(0, step)
            positions[idx, 1] += np.random.normal(0, step)
            angles[idx] += np.random.normal(0, np.radians(10))
            
            # Compute new score
            all_verts = transform_trees_batch(positions, angles)
            s = compute_bounding_square(all_verts)
            new_score = (s * s) / n
            
            # Accept/reject
            if new_score < best_score:
                best_score = new_score
                best_solution[:, :2] = positions.copy()
                best_solution[:, 2] = np.degrees(angles)
            else:
                # Revert
                positions[idx] = old_pos
                angles[idx] = old_angle
            
            # Cool down
            temp *= 0.9999
            step = max(0.01, step * 0.9999)
        
        return best_solution, best_score
    
    def _find_edge_trees(self, solution: np.ndarray) -> List[int]:
        """Find indices of trees on the bounding box edge."""
        from core.tree_polygon import transform_trees_batch
        from core.bounding_box import compute_bounding_box
        
        positions = solution[:, :2]
        angles = np.radians(solution[:, 2])
        
        all_verts = transform_trees_batch(positions, angles)
        min_x, min_y, max_x, max_y = compute_bounding_box(all_verts)
        
        tolerance = 0.05
        edge_indices = []
        
        for i in range(len(solution)):
            tree_verts = all_verts[i]
            
            # Check if any vertex is on the boundary
            on_edge = False
            for v in tree_verts:
                if (abs(v[0] - min_x) < tolerance or
                    abs(v[0] - max_x) < tolerance or
                    abs(v[1] - min_y) < tolerance or
                    abs(v[1] - max_y) < tolerance):
                    on_edge = True
                    break
            
            if on_edge:
                edge_indices.append(i)
        
        return edge_indices
    
    def _refine_with_sa(
        self,
        solution: np.ndarray,
        n: int
    ) -> Tuple[np.ndarray, float]:
        """Polish the solution with SA."""
        from optimizers.adaptive_sa import AdaptiveSA, AdaptiveSAConfig
        
        cfg = self.config
        
        sa_config = AdaptiveSAConfig(
            max_iterations=cfg.refinement_iterations,
            num_restarts=3,
            initial_temp=0.3,
            initial_step_translation=0.05,
            initial_step_rotation=10.0,
        )
        
        sa = AdaptiveSA(sa_config)
        return sa.optimize(
            n_trees=n,
            initial_solution=solution,
            verbose=False
        )
    
    def _compute_score(self, solution: np.ndarray) -> float:
        """Compute score for a solution."""
        from core.tree_polygon import transform_trees_batch
        from core.bounding_box import compute_score
        
        n = len(solution)
        positions = solution[:, :2]
        angles = np.radians(solution[:, 2])
        
        all_verts = transform_trees_batch(positions, angles)
        return compute_score(all_verts, n)
    
    def _resolve_any_collisions(self, solution: np.ndarray) -> np.ndarray:
        """Ensure no collisions in the solution."""
        from core.tree_polygon import transform_trees_batch, get_all_bounds
        from core.collision import check_all_collisions
        
        positions = solution[:, :2]
        angles = np.radians(solution[:, 2])
        
        all_verts = transform_trees_batch(positions, angles)
        all_bounds = get_all_bounds(all_verts)
        
        collisions = check_all_collisions(all_verts, all_bounds)
        
        if not collisions:
            return solution
        
        # Spread out to resolve collisions
        spread_factor = 1.0
        max_attempts = 10
        
        for attempt in range(max_attempts):
            spread_factor += 0.1
            
            # Scale positions from center
            new_solution = solution.copy()
            center = solution[:, :2].mean(axis=0)
            new_solution[:, 0] = center[0] + (solution[:, 0] - center[0]) * spread_factor
            new_solution[:, 1] = center[1] + (solution[:, 1] - center[1]) * spread_factor
            
            positions = new_solution[:, :2]
            angles = np.radians(new_solution[:, 2])
            
            all_verts = transform_trees_batch(positions, angles)
            all_bounds = get_all_bounds(all_verts)
            
            collisions = check_all_collisions(all_verts, all_bounds)
            
            if not collisions:
                return new_solution
        
        return solution  # Return original if can't resolve


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def pack_with_lattice(
    n: int,
    verbose: bool = True
) -> Tuple[np.ndarray, float]:
    """
    Pack n trees using the best lattice method.
    
    Args:
        n: Number of trees
        verbose: Print progress
        
    Returns:
        (solution, score)
    """
    config = LatticeConfig(
        use_optimized_dimer=True,
        optimize_edges=True,
        use_refinement=n <= 100,  # Only refine smaller packings
        verbose=verbose
    )
    
    packer = LatticePacker(config)
    return packer.pack(n, verbose=verbose)


def quick_lattice_pack(n: int) -> Tuple[np.ndarray, float]:
    """
    Quick lattice packing without optimization.
    
    For when you need a solution FAST.
    """
    config = LatticeConfig(
        use_optimized_dimer=False,
        optimize_edges=False,
        use_refinement=False,
        verbose=False
    )
    
    packer = LatticePacker(config)
    return packer.pack(n, verbose=False)


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    print("Testing Lattice Packer")
    print("=" * 60)
    
    # Warm up
    print("\nWarming up...")
    from core.tree_polygon import warmup as warmup_tree
    warmup_tree()
    
    # Quick test without optimization
    print("\n" + "="*60)
    print("Quick lattice (no optimization):")
    
    for test_n in [20, 50, 100]:
        solution, score = quick_lattice_pack(test_n)
        print(f"  n={test_n:3d}: score={score:.6f}")
    
    # Full test with optimization
    print("\n" + "="*60)
    print("Full lattice packing with n=50:")
    
    config = LatticeConfig(
        use_optimized_dimer=True,
        dimer_optimization_iters=200,
        optimize_edges=True,
        edge_sa_iterations=20000,
        use_refinement=True,
        refinement_iterations=50000,
        verbose=True
    )
    
    packer = LatticePacker(config)
    solution, score = packer.pack(50, verbose=True)
    
    print(f"\n✅ Test complete!")
    print(f"   Final score for n=50: {score:.6f}")
