"""
Christmas Tree Polygon - Core geometry with Numba-accelerated operations.

This module defines the Christmas tree shape and provides high-performance
transformation functions using Numba JIT compilation.

The tree is a 16-vertex concave polygon with dimensions:
- Height: 1.0 units (from trunk bottom at -0.2 to tip at 0.8)
- Max Width: 0.7 units (at base)
- Area: ~0.285 square units
"""

import numpy as np
from numba import njit, prange, float64
from numba.types import Array
from typing import Tuple
import math


# =============================================================================
# TREE POLYGON DEFINITION
# =============================================================================

# Tree dimensions (from competition specification)
TRUNK_WIDTH = 0.15
TRUNK_HEIGHT = 0.2
BASE_WIDTH = 0.7
MID_WIDTH = 0.4
TOP_WIDTH = 0.25
TIP_Y = 0.8

# Christmas tree vertices (16 points, counter-clockwise from tip)
# Centered at origin for rotation
TREE_VERTICES = np.array([
    # Tip
    [0.0, 0.8],
    
    # Right side - top tier
    [TOP_WIDTH / 2, 0.5],           # 0.125
    [TOP_WIDTH / 4, 0.5],           # 0.0625 (notch)
    
    # Right side - middle tier  
    [MID_WIDTH / 2, 0.25],          # 0.2
    [MID_WIDTH / 4, 0.25],          # 0.1 (notch)
    
    # Right side - base tier
    [BASE_WIDTH / 2, 0.0],          # 0.35
    
    # Right trunk
    [TRUNK_WIDTH / 2, 0.0],         # 0.075
    [TRUNK_WIDTH / 2, -TRUNK_HEIGHT],  # 0.075, -0.2
    
    # Left trunk
    [-TRUNK_WIDTH / 2, -TRUNK_HEIGHT],  # -0.075, -0.2
    [-TRUNK_WIDTH / 2, 0.0],            # -0.075
    
    # Left side - base tier
    [-BASE_WIDTH / 2, 0.0],             # -0.35
    
    # Left side - middle tier
    [-MID_WIDTH / 4, 0.25],             # -0.1 (notch)
    [-MID_WIDTH / 2, 0.25],             # -0.2
    
    # Left side - top tier
    [-TOP_WIDTH / 4, 0.5],              # -0.0625 (notch)
    [-TOP_WIDTH / 2, 0.5],              # -0.125
], dtype=np.float64)

# Number of vertices
NUM_VERTICES = len(TREE_VERTICES)

# Precompute tree area (for reference)
@njit(cache=True)
def _polygon_area(vertices: np.ndarray) -> float:
    """Compute polygon area using shoelace formula."""
    n = len(vertices)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i, 0] * vertices[j, 1]
        area -= vertices[j, 0] * vertices[i, 1]
    return abs(area) / 2.0

TREE_AREA = _polygon_area(TREE_VERTICES)


# =============================================================================
# NUMBA-ACCELERATED TRANSFORMATIONS
# =============================================================================

@njit(cache=True, fastmath=True)
def rotate_vertices(vertices: np.ndarray, angle_rad: float) -> np.ndarray:
    """
    Rotate vertices around origin by angle (in radians).
    
    Args:
        vertices: (N, 2) array of vertex coordinates
        angle_rad: Rotation angle in radians
        
    Returns:
        Rotated vertices array (N, 2)
    """
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    
    n = len(vertices)
    rotated = np.empty((n, 2), dtype=np.float64)
    
    for i in range(n):
        x = vertices[i, 0]
        y = vertices[i, 1]
        rotated[i, 0] = x * cos_a - y * sin_a
        rotated[i, 1] = x * sin_a + y * cos_a
    
    return rotated


@njit(cache=True, fastmath=True)
def translate_vertices(vertices: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    Translate vertices by (dx, dy).
    
    Args:
        vertices: (N, 2) array of vertex coordinates
        dx: Translation in x direction
        dy: Translation in y direction
        
    Returns:
        Translated vertices array (N, 2)
    """
    n = len(vertices)
    translated = np.empty((n, 2), dtype=np.float64)
    
    for i in range(n):
        translated[i, 0] = vertices[i, 0] + dx
        translated[i, 1] = vertices[i, 1] + dy
    
    return translated


@njit(cache=True, fastmath=True)
def transform_tree(x: float, y: float, angle_rad: float) -> np.ndarray:
    """
    Get transformed tree vertices for position (x, y) and rotation angle.
    
    Args:
        x: Center x position
        y: Center y position  
        angle_rad: Rotation angle in radians
        
    Returns:
        Transformed vertices (16, 2)
    """
    # Rotate first, then translate
    rotated = rotate_vertices(TREE_VERTICES, angle_rad)
    return translate_vertices(rotated, x, y)


@njit(cache=True, fastmath=True, parallel=True)
def transform_trees_batch(
    positions: np.ndarray, 
    angles: np.ndarray
) -> np.ndarray:
    """
    Transform multiple trees in parallel.
    
    Args:
        positions: (n, 2) array of (x, y) positions
        angles: (n,) array of rotation angles in radians
        
    Returns:
        (n, 16, 2) array of transformed vertices for all trees
    """
    n = len(positions)
    result = np.empty((n, NUM_VERTICES, 2), dtype=np.float64)
    
    for i in prange(n):
        result[i] = transform_tree(
            positions[i, 0], 
            positions[i, 1], 
            angles[i]
        )
    
    return result


@njit(cache=True, fastmath=True)
def get_bounds(vertices: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Get axis-aligned bounding box for vertices.
    
    Returns:
        (min_x, min_y, max_x, max_y)
    """
    min_x = vertices[0, 0]
    max_x = vertices[0, 0]
    min_y = vertices[0, 1]
    max_y = vertices[0, 1]
    
    for i in range(1, len(vertices)):
        x = vertices[i, 0]
        y = vertices[i, 1]
        
        if x < min_x:
            min_x = x
        elif x > max_x:
            max_x = x
            
        if y < min_y:
            min_y = y
        elif y > max_y:
            max_y = y
    
    return min_x, min_y, max_x, max_y


@njit(cache=True, fastmath=True, parallel=True)
def get_all_bounds(all_vertices: np.ndarray) -> np.ndarray:
    """
    Get bounding boxes for all trees.
    
    Args:
        all_vertices: (n, 16, 2) array
        
    Returns:
        (n, 4) array of (min_x, min_y, max_x, max_y)
    """
    n = len(all_vertices)
    bounds = np.empty((n, 4), dtype=np.float64)
    
    for i in prange(n):
        min_x, min_y, max_x, max_y = get_bounds(all_vertices[i])
        bounds[i, 0] = min_x
        bounds[i, 1] = min_y
        bounds[i, 2] = max_x
        bounds[i, 3] = max_y
    
    return bounds


# =============================================================================
# CHRISTMAS TREE CLASS
# =============================================================================

class ChristmasTree:
    """
    High-level Christmas tree object with position, rotation, and caching.
    
    Attributes:
        x: X position of tree center
        y: Y position of tree center
        angle_deg: Rotation angle in degrees
        
    Properties:
        vertices: Transformed vertices (cached)
        bounds: Bounding box (min_x, min_y, max_x, max_y)
    """
    
    __slots__ = ['x', 'y', '_angle_rad', '_vertices', '_bounds', '_dirty']
    
    def __init__(self, x: float = 0.0, y: float = 0.0, angle_deg: float = 0.0):
        self.x = x
        self.y = y
        self._angle_rad = np.radians(angle_deg)
        self._vertices = None
        self._bounds = None
        self._dirty = True
    
    @property
    def angle_deg(self) -> float:
        """Rotation angle in degrees."""
        return np.degrees(self._angle_rad)
    
    @angle_deg.setter
    def angle_deg(self, value: float):
        self._angle_rad = np.radians(value)
        self._dirty = True
    
    @property
    def angle_rad(self) -> float:
        """Rotation angle in radians."""
        return self._angle_rad
    
    @angle_rad.setter
    def angle_rad(self, value: float):
        self._angle_rad = value
        self._dirty = True
    
    def set_position(self, x: float, y: float):
        """Set tree center position."""
        self.x = x
        self.y = y
        self._dirty = True
    
    def move(self, dx: float, dy: float):
        """Move tree by delta."""
        self.x += dx
        self.y += dy
        self._dirty = True
    
    def rotate(self, delta_deg: float):
        """Rotate tree by delta degrees."""
        self._angle_rad += np.radians(delta_deg)
        self._dirty = True
    
    @property
    def vertices(self) -> np.ndarray:
        """Get transformed vertices (16, 2). Cached."""
        if self._dirty:
            self._update()
        return self._vertices
    
    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """Get bounding box (min_x, min_y, max_x, max_y). Cached."""
        if self._dirty:
            self._update()
        return self._bounds
    
    def _update(self):
        """Recompute vertices and bounds."""
        self._vertices = transform_tree(self.x, self.y, self._angle_rad)
        self._bounds = get_bounds(self._vertices)
        self._dirty = False
    
    def to_tuple(self) -> Tuple[float, float, float]:
        """Return (x, y, angle_deg) for submission."""
        return (self.x, self.y, np.degrees(self._angle_rad))
    
    def to_array(self) -> np.ndarray:
        """Return [x, y, angle_deg] as numpy array."""
        return np.array([self.x, self.y, np.degrees(self._angle_rad)])
    
    def copy(self) -> 'ChristmasTree':
        """Create a copy of this tree."""
        return ChristmasTree(self.x, self.y, np.degrees(self._angle_rad))
    
    def __repr__(self) -> str:
        return f"ChristmasTree(x={self.x:.4f}, y={self.y:.4f}, angle={self.angle_deg:.2f}°)"


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_trees_from_solution(solution: np.ndarray) -> list:
    """
    Create list of ChristmasTree objects from solution array.
    
    Args:
        solution: (n, 3) array of [x, y, angle_deg]
        
    Returns:
        List of ChristmasTree objects
    """
    return [
        ChristmasTree(x=row[0], y=row[1], angle_deg=row[2])
        for row in solution
    ]


def solution_to_arrays(solution: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert solution array to positions and angles.
    
    Args:
        solution: (n, 3) array of [x, y, angle_deg]
        
    Returns:
        positions: (n, 2) array
        angles_rad: (n,) array
    """
    positions = solution[:, :2].copy()
    angles_rad = np.radians(solution[:, 2])
    return positions, angles_rad


def arrays_to_solution(positions: np.ndarray, angles_rad: np.ndarray) -> np.ndarray:
    """
    Convert positions and angles to solution array.
    
    Args:
        positions: (n, 2) array
        angles_rad: (n,) array
        
    Returns:
        solution: (n, 3) array of [x, y, angle_deg]
    """
    n = len(positions)
    solution = np.empty((n, 3), dtype=np.float64)
    solution[:, :2] = positions
    solution[:, 2] = np.degrees(angles_rad)
    return solution


# =============================================================================
# WARM-UP JIT COMPILATION
# =============================================================================

def warmup():
    """Warm up JIT compilation by calling all functions once."""
    # Create test data
    pos = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    angles = np.array([0.0, np.pi/4], dtype=np.float64)
    
    # Call all functions
    _ = rotate_vertices(TREE_VERTICES, 0.5)
    _ = translate_vertices(TREE_VERTICES, 1.0, 1.0)
    _ = transform_tree(0.0, 0.0, 0.0)
    _ = transform_trees_batch(pos, angles)
    _ = get_bounds(TREE_VERTICES)
    _ = get_all_bounds(np.stack([TREE_VERTICES, TREE_VERTICES]))
    
    print("JIT warmup complete for tree_polygon module")


if __name__ == "__main__":
    # Run warmup when module is executed directly
    warmup()
    
    # Demo
    print(f"\nTree vertices ({NUM_VERTICES} points):")
    print(TREE_VERTICES)
    print(f"\nTree area: {TREE_AREA:.4f} sq units")
    
    # Test transformation
    tree = ChristmasTree(x=1.0, y=2.0, angle_deg=45.0)
    print(f"\n{tree}")
    print(f"Bounds: {tree.bounds}")
