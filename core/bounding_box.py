"""
Bounding Box Computation - Score calculation for the competition.

The competition scores solutions by: sum of (s²/n) for all n from 1 to 200
where s is the side of the smallest square containing all trees.
"""

import numpy as np
from numba import njit, prange
from typing import Tuple
import math


@njit(cache=True, fastmath=True)
def compute_bounding_box(all_vertices: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Compute axis-aligned bounding box of all vertices.
    
    Args:
        all_vertices: (n, num_verts, 2) array of all tree vertices
        
    Returns:
        (min_x, min_y, max_x, max_y)
    """
    min_x = math.inf
    max_x = -math.inf
    min_y = math.inf
    max_y = -math.inf
    
    n_trees = len(all_vertices)
    n_verts = all_vertices.shape[1]
    
    for i in range(n_trees):
        for j in range(n_verts):
            x = all_vertices[i, j, 0]
            y = all_vertices[i, j, 1]
            
            if x < min_x:
                min_x = x
            if x > max_x:
                max_x = x
            if y < min_y:
                min_y = y
            if y > max_y:
                max_y = y
    
    return min_x, min_y, max_x, max_y


@njit(cache=True, fastmath=True)
def compute_bounding_square(all_vertices: np.ndarray) -> float:
    """
    Compute side length of minimum axis-aligned bounding square.
    
    The square must fully contain all trees, so we take the max
    of width and height.
    
    Args:
        all_vertices: (n, 16, 2) array of all tree vertices
        
    Returns:
        Side length of bounding square
    """
    min_x, min_y, max_x, max_y = compute_bounding_box(all_vertices)
    
    width = max_x - min_x
    height = max_y - min_y
    
    return max(width, height)


@njit(cache=True, fastmath=True)
def compute_score(all_vertices: np.ndarray, n: int) -> float:
    """
    Compute competition score for a single configuration.
    
    Score = s² / n
    
    where s is the bounding square side length and n is number of trees.
    
    Args:
        all_vertices: (n, 16, 2) array
        n: Number of trees
        
    Returns:
        Score contribution for this configuration
    """
    s = compute_bounding_square(all_vertices)
    return (s * s) / n


@njit(cache=True, fastmath=True)
def compute_bounding_square_from_positions(
    positions: np.ndarray,
    angles_rad: np.ndarray,
    tree_vertices: np.ndarray
) -> float:
    """
    Compute bounding square from positions and angles without creating full vertex array.
    
    More memory efficient for frequent score calculations.
    """
    n = len(positions)
    num_verts = len(tree_vertices)
    
    min_x = math.inf
    max_x = -math.inf
    min_y = math.inf
    max_y = -math.inf
    
    for i in range(n):
        cx = positions[i, 0]
        cy = positions[i, 1]
        angle = angles_rad[i]
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        
        for j in range(num_verts):
            # Rotate and translate
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
    
    width = max_x - min_x
    height = max_y - min_y
    
    return max(width, height)


def get_bounding_box(solution: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Get bounding box from solution array.
    
    Args:
        solution: (n, 3) array of [x, y, angle_deg]
        
    Returns:
        (min_x, min_y, max_x, max_y)
    """
    from core.tree_polygon import transform_trees_batch
    
    positions = solution[:, :2]
    angles_rad = np.radians(solution[:, 2])
    
    all_vertices = transform_trees_batch(positions, angles_rad)
    return compute_bounding_box(all_vertices)


def get_score(solution: np.ndarray) -> float:
    """
    Compute score for a solution.
    
    Args:
        solution: (n, 3) array of [x, y, angle_deg]
        
    Returns:
        Score (s²/n)
    """
    from core.tree_polygon import transform_trees_batch
    
    n = len(solution)
    positions = solution[:, :2]
    angles_rad = np.radians(solution[:, 2])
    
    all_vertices = transform_trees_batch(positions, angles_rad)
    return compute_score(all_vertices, n)


def center_solution(solution: np.ndarray) -> np.ndarray:
    """
    Center solution so bounding box is centered at origin.
    
    This doesn't change the score but makes visualization nicer.
    
    Args:
        solution: (n, 3) array
        
    Returns:
        Centered solution
    """
    from core.tree_polygon import transform_trees_batch
    
    positions = solution[:, :2]
    angles_rad = np.radians(solution[:, 2])
    
    all_vertices = transform_trees_batch(positions, angles_rad)
    min_x, min_y, max_x, max_y = compute_bounding_box(all_vertices)
    
    # Compute center offset
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    
    # Create centered solution
    centered = solution.copy()
    centered[:, 0] -= center_x
    centered[:, 1] -= center_y
    
    return centered


def compute_total_score(results: dict) -> float:
    """
    Compute total score from results dictionary.
    
    Args:
        results: Dict mapping n -> OptimizationResult or score
        
    Returns:
        Total score across all configurations
    """
    total = 0.0
    for n in range(1, 201):
        if n in results:
            result = results[n]
            if hasattr(result, 'score'):
                total += result.score
            else:
                total += result  # Assume it's the score directly
    return total


# =============================================================================
# WARMUP
# =============================================================================

def warmup():
    """Warm up JIT compilation."""
    from core.tree_polygon import TREE_VERTICES, NUM_VERTICES
    
    # Create dummy data - use actual vertex count from TREE_VERTICES
    n_verts = len(TREE_VERTICES)
    verts = np.zeros((5, n_verts, 2), dtype=np.float64)
    for i in range(5):
        verts[i] = TREE_VERTICES + np.array([i * 1.5, 0])
    
    _ = compute_bounding_box(verts)
    _ = compute_bounding_square(verts)
    _ = compute_score(verts, 5)
    
    pos = np.array([[0.0, 0.0], [1.5, 0.0]], dtype=np.float64)
    angles = np.array([0.0, 0.0], dtype=np.float64)
    _ = compute_bounding_square_from_positions(pos, angles, TREE_VERTICES)
    
    print("JIT warmup complete for bounding_box module")


if __name__ == "__main__":
    warmup()
    
    # Demo
    from tree_polygon import transform_trees_batch, TREE_VERTICES
    
    # Create simple arrangement
    n = 4
    solution = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.2, 0.0],
        [1.0, 1.2, 0.0],
    ])
    
    score = get_score(solution)
    bbox = get_bounding_box(solution)
    
    print(f"\n{n} trees in grid:")
    print(f"  Bounding box: {bbox}")
    print(f"  Square side: {max(bbox[2]-bbox[0], bbox[3]-bbox[1]):.4f}")
    print(f"  Score (s²/n): {score:.6f}")
