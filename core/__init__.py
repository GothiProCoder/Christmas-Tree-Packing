"""
Core module - Geometry, collision detection, and fundamental operations.
"""

from .tree_polygon import (
    TREE_VERTICES,
    ChristmasTree,
    transform_tree,
    transform_trees_batch,
    rotate_vertices,
    translate_vertices,
)

from .collision import (
    polygons_overlap,
    bounds_overlap,
    check_all_collisions,
    check_any_collision,
)

from .bounding_box import (
    compute_bounding_square,
    compute_score,
    get_bounding_box,
)

__all__ = [
    'TREE_VERTICES',
    'ChristmasTree', 
    'transform_tree',
    'transform_trees_batch',
    'rotate_vertices',
    'translate_vertices',
    'polygons_overlap',
    'bounds_overlap',
    'check_all_collisions',
    'check_any_collision',
    'compute_bounding_square',
    'compute_score',
    'get_bounding_box',
]
