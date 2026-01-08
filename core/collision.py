"""
Collision Detection - Fast polygon intersection using SAT and Shapely.

This module provides multiple collision detection methods:
1. SAT (Separating Axis Theorem) - Fast but for convex polygons
2. Shapely-based - Accurate for concave polygons (like our tree)
3. AABB pre-check - Ultra-fast bounding box filter

For the concave Christmas tree, we use AABB pre-check + Shapely for accuracy.
"""

import numpy as np
from numba import njit, prange
from typing import Tuple, List, Optional
import math


# =============================================================================
# AXIS-ALIGNED BOUNDING BOX (AABB) CHECKS
# =============================================================================

@njit(cache=True, fastmath=True)
def bounds_overlap(
    b1_min_x: float, b1_min_y: float, b1_max_x: float, b1_max_y: float,
    b2_min_x: float, b2_min_y: float, b2_max_x: float, b2_max_y: float
) -> bool:
    """
    Check if two AABBs overlap.
    
    Returns True if overlapping, False if separated.
    """
    return not (
        b1_max_x < b2_min_x or b2_max_x < b1_min_x or
        b1_max_y < b2_min_y or b2_max_y < b1_min_y
    )


@njit(cache=True, fastmath=True)
def bounds_overlap_array(b1: np.ndarray, b2: np.ndarray) -> bool:
    """Check if two AABB arrays [min_x, min_y, max_x, max_y] overlap."""
    return bounds_overlap(
        b1[0], b1[1], b1[2], b1[3],
        b2[0], b2[1], b2[2], b2[3]
    )


# =============================================================================
# SEPARATING AXIS THEOREM (SAT) - For reference/convex shapes
# =============================================================================

@njit(cache=True, fastmath=True)
def project_polygon(vertices: np.ndarray, axis_x: float, axis_y: float) -> Tuple[float, float]:
    """
    Project polygon onto axis, return (min, max) projection.
    """
    min_proj = math.inf
    max_proj = -math.inf
    
    for i in range(len(vertices)):
        proj = vertices[i, 0] * axis_x + vertices[i, 1] * axis_y
        if proj < min_proj:
            min_proj = proj
        if proj > max_proj:
            max_proj = proj
    
    return min_proj, max_proj


@njit(cache=True, fastmath=True)
def sat_overlap(verts1: np.ndarray, verts2: np.ndarray) -> bool:
    """
    Check if two convex polygons overlap using SAT.
    
    NOTE: This works correctly only for CONVEX polygons.
    For the concave Christmas tree, use shapely_overlap instead.
    
    Returns True if overlapping, False if separated.
    """
    n1 = len(verts1)
    n2 = len(verts2)
    
    # Check all edges from polygon 1
    for i in range(n1):
        # Edge vector
        edge_x = verts1[(i + 1) % n1, 0] - verts1[i, 0]
        edge_y = verts1[(i + 1) % n1, 1] - verts1[i, 1]
        
        # Perpendicular (normal) - potential separating axis
        axis_x = -edge_y
        axis_y = edge_x
        
        # Normalize
        length = math.sqrt(axis_x * axis_x + axis_y * axis_y)
        if length > 1e-10:
            axis_x /= length
            axis_y /= length
        
        # Project both polygons
        min1, max1 = project_polygon(verts1, axis_x, axis_y)
        min2, max2 = project_polygon(verts2, axis_x, axis_y)
        
        # Check for gap (separation)
        if max1 < min2 or max2 < min1:
            return False  # Found separating axis
    
    # Check all edges from polygon 2
    for i in range(n2):
        edge_x = verts2[(i + 1) % n2, 0] - verts2[i, 0]
        edge_y = verts2[(i + 1) % n2, 1] - verts2[i, 1]
        
        axis_x = -edge_y
        axis_y = edge_x
        
        length = math.sqrt(axis_x * axis_x + axis_y * axis_y)
        if length > 1e-10:
            axis_x /= length
            axis_y /= length
        
        min1, max1 = project_polygon(verts1, axis_x, axis_y)
        min2, max2 = project_polygon(verts2, axis_x, axis_y)
        
        if max1 < min2 or max2 < min1:
            return False
    
    return True  # No separating axis found = overlapping


# =============================================================================
# SHAPELY-BASED COLLISION (Accurate for concave polygons)
# =============================================================================

def shapely_overlap(verts1: np.ndarray, verts2: np.ndarray) -> bool:
    """
    Check if two polygons overlap using Shapely.
    
    This handles concave polygons correctly.
    Returns True if interiors intersect (not just touching).
    """
    try:
        from shapely.geometry import Polygon
        from shapely.validation import make_valid
        
        p1 = Polygon(verts1)
        p2 = Polygon(verts2)
        
        # Make valid if needed
        if not p1.is_valid:
            p1 = make_valid(p1)
        if not p2.is_valid:
            p2 = make_valid(p2)
        
        # Check for intersection (excluding just touching)
        return p1.intersects(p2) and not p1.touches(p2)
    except ImportError:
        # Shapely not available - fallback to SAT (less accurate for concave)
        return sat_overlap(verts1, verts2)
    except Exception:
        # Other error - assume overlap to be safe
        return True


def shapely_overlap_interior(verts1: np.ndarray, verts2: np.ndarray) -> bool:
    """
    Check if polygon interiors overlap (stricter check).
    """
    try:
        from shapely.geometry import Polygon
        
        p1 = Polygon(verts1)
        p2 = Polygon(verts2)
        return p1.overlaps(p2) or p1.contains(p2) or p2.contains(p1)
    except ImportError:
        # Shapely not available - fallback to SAT
        return sat_overlap(verts1, verts2)
    except Exception:
        return True


# =============================================================================
# MAIN COLLISION CHECKING FUNCTIONS
# =============================================================================

def polygons_overlap(verts1: np.ndarray, verts2: np.ndarray, use_shapely: bool = True) -> bool:
    """
    Check if two polygons overlap.
    
    Args:
        verts1: First polygon vertices (N, 2)
        verts2: Second polygon vertices (M, 2)
        use_shapely: Use Shapely for accurate concave handling
        
    Returns:
        True if overlapping, False otherwise
    """
    if use_shapely:
        return shapely_overlap(verts1, verts2)
    else:
        return sat_overlap(verts1, verts2)


def check_any_collision(
    all_vertices: np.ndarray,
    all_bounds: np.ndarray,
    use_shapely: bool = True
) -> bool:
    """
    Check if ANY pair of trees collides.
    
    Uses AABB pre-check for speed, then detailed check for candidates.
    
    Args:
        all_vertices: (n, 16, 2) array of all tree vertices
        all_bounds: (n, 4) array of bounding boxes
        use_shapely: Use Shapely for detailed check
        
    Returns:
        True if any collision exists, False otherwise
    """
    n = len(all_vertices)
    
    for i in range(n):
        for j in range(i + 1, n):
            # Quick AABB check
            if bounds_overlap_array(all_bounds[i], all_bounds[j]):
                # Detailed polygon check
                if polygons_overlap(all_vertices[i], all_vertices[j], use_shapely):
                    return True
    
    return False


def check_all_collisions(
    all_vertices: np.ndarray,
    all_bounds: np.ndarray,
    use_shapely: bool = True
) -> List[Tuple[int, int]]:
    """
    Find ALL colliding pairs.
    
    Args:
        all_vertices: (n, 16, 2) array
        all_bounds: (n, 4) array
        use_shapely: Use Shapely for accuracy
        
    Returns:
        List of (i, j) tuples for colliding pairs
    """
    n = len(all_vertices)
    collisions = []
    
    for i in range(n):
        for j in range(i + 1, n):
            if bounds_overlap_array(all_bounds[i], all_bounds[j]):
                if polygons_overlap(all_vertices[i], all_vertices[j], use_shapely):
                    collisions.append((i, j))
    
    return collisions


def check_tree_collides_with_others(
    tree_idx: int,
    all_vertices: np.ndarray,
    all_bounds: np.ndarray,
    use_shapely: bool = True
) -> bool:
    """
    Check if a specific tree collides with any other tree.
    
    Useful for checking a single moved tree without full O(n²) check.
    """
    n = len(all_vertices)
    tree_verts = all_vertices[tree_idx]
    tree_bounds = all_bounds[tree_idx]
    
    for j in range(n):
        if j == tree_idx:
            continue
        
        if bounds_overlap_array(tree_bounds, all_bounds[j]):
            if polygons_overlap(tree_verts, all_vertices[j], use_shapely):
                return True
    
    return False


# =============================================================================
# COLLISION CHECKER CLASS (for use in optimizers)
# =============================================================================

class CollisionChecker:
    """
    Collision checker with caching and Shapely spatial index.
    
    Provides efficient collision detection for optimization loops.
    """
    
    def __init__(self, use_shapely: bool = True, use_strtree: bool = True):
        self.use_shapely = use_shapely
        self.use_strtree = use_strtree
        self._strtree = None
        self._polygons = None
    
    def build_index(self, all_vertices: np.ndarray):
        """Build Shapely STRtree spatial index for fast queries."""
        if self.use_strtree:
            from shapely.geometry import Polygon
            from shapely import STRtree
            
            self._polygons = [Polygon(verts) for verts in all_vertices]
            self._strtree = STRtree(self._polygons)
    
    def check_any_collision_indexed(self, all_vertices: np.ndarray) -> bool:
        """Check collisions using spatial index."""
        from shapely.geometry import Polygon
        
        if self._strtree is None:
            self.build_index(all_vertices)
        
        n = len(all_vertices)
        for i in range(n):
            poly = Polygon(all_vertices[i])
            # Query nearby polygons
            candidates = self._strtree.query(poly)
            for j in candidates:
                if j <= i:
                    continue
                if poly.intersects(self._polygons[j]) and not poly.touches(self._polygons[j]):
                    return True
        
        return False
    
    def check(
        self,
        positions: np.ndarray,
        angles_rad: np.ndarray
    ) -> bool:
        """
        Check if current configuration has any collisions.
        
        This is the main method called by optimizers.
        """
        from core.tree_polygon import transform_trees_batch, get_all_bounds
        
        all_vertices = transform_trees_batch(positions, angles_rad)
        all_bounds = get_all_bounds(all_vertices)
        
        return check_any_collision(all_vertices, all_bounds, self.use_shapely)


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_solution(solution: np.ndarray, tolerance: float = 0.0) -> Tuple[bool, List[Tuple[int, int]]]:
    """
    Validate a solution for collisions.
    
    Args:
        solution: (n, 3) array of [x, y, angle_deg]
        tolerance: Buffer distance (0 = strict, >0 = allow small gaps)
        
    Returns:
        (is_valid, list_of_collisions)
    """
    from core.tree_polygon import transform_trees_batch, get_all_bounds
    
    positions = solution[:, :2]
    angles_rad = np.radians(solution[:, 2])
    
    all_vertices = transform_trees_batch(positions, angles_rad)
    all_bounds = get_all_bounds(all_vertices)
    
    collisions = check_all_collisions(all_vertices, all_bounds, use_shapely=True)
    
    return len(collisions) == 0, collisions


# =============================================================================
# WARMUP
# =============================================================================

def warmup():
    """Warm up JIT compilation."""
    verts = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0]
    ], dtype=np.float64)
    
    _ = bounds_overlap(0, 0, 1, 1, 0.5, 0.5, 1.5, 1.5)
    _ = bounds_overlap_array(np.array([0, 0, 1, 1.0]), np.array([0.5, 0.5, 1.5, 1.5]))
    _ = project_polygon(verts, 1.0, 0.0)
    _ = sat_overlap(verts, verts)
    
    print("JIT warmup complete for collision module")


if __name__ == "__main__":
    warmup()
    
    # Test with actual tree
    from tree_polygon import TREE_VERTICES, transform_tree
    
    # Two trees that should not collide
    tree1 = transform_tree(0.0, 0.0, 0.0)
    tree2 = transform_tree(2.0, 0.0, 0.0)
    
    print(f"\nTrees at (0,0) and (2,0): overlap={shapely_overlap(tree1, tree2)}")
    
    # Two trees that should collide
    tree3 = transform_tree(0.5, 0.0, 0.0)
    print(f"Trees at (0,0) and (0.5,0): overlap={shapely_overlap(tree1, tree3)}")
