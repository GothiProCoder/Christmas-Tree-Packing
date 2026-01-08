"""
Visualization - Plotting trees, solutions, and optimization progress.

Provides tools for debugging and understanding packing configurations.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
from typing import Optional, List, Tuple
import os


def plot_tree_shape(ax=None, color='green', alpha=0.7):
    """
    Plot the base Christmas tree shape.
    
    Args:
        ax: Matplotlib axes (creates new if None)
        color: Fill color
        alpha: Transparency
    """
    from core.tree_polygon import TREE_VERTICES
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 8))
    
    # Close the polygon
    verts = np.vstack([TREE_VERTICES, TREE_VERTICES[0]])
    
    ax.fill(verts[:, 0], verts[:, 1], color=color, alpha=alpha, edgecolor='darkgreen', linewidth=2)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title('Christmas Tree Shape')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    return ax


def plot_trees(
    solution: np.ndarray,
    ax=None,
    title: str = None,
    show_bounds: bool = True,
    show_score: bool = True,
    highlight_collisions: bool = True,
    tree_colors: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 10)
):
    """
    Plot a tree packing solution.
    
    Args:
        solution: (n, 3) array of [x, y, angle_deg]
        ax: Matplotlib axes
        title: Plot title
        show_bounds: Draw bounding square
        show_score: Display score in title
        highlight_collisions: Color colliding trees red
        tree_colors: Custom colors for each tree
        figsize: Figure size if creating new figure
        
    Returns:
        ax: Matplotlib axes
    """
    from core.tree_polygon import transform_trees_batch
    from core.bounding_box import compute_bounding_box, compute_score
    from core.collision import check_all_collisions
    
    n = len(solution)
    positions = solution[:, :2]
    angles_rad = np.radians(solution[:, 2])
    
    # Get all vertices
    all_vertices = transform_trees_batch(positions, angles_rad)
    
    # Find collisions
    colliding_trees = set()
    if highlight_collisions:
        from core.tree_polygon import get_all_bounds
        all_bounds = get_all_bounds(all_vertices)
        collisions = check_all_collisions(all_vertices, all_bounds)
        for i, j in collisions:
            colliding_trees.add(i)
            colliding_trees.add(j)
    
    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Create polygon patches
    patches = []
    colors = []
    
    for i in range(n):
        verts = all_vertices[i]
        poly = MplPolygon(verts, closed=True)
        patches.append(poly)
        
        if tree_colors is not None:
            colors.append(tree_colors[i])
        elif i in colliding_trees:
            colors.append('red')
        else:
            # Gradient green based on index
            green = 0.4 + 0.4 * (i / max(n-1, 1))
            colors.append((0.1, green, 0.1))
    
    # Add patches to plot
    collection = PatchCollection(
        patches, 
        facecolors=colors,
        edgecolors='darkgreen',
        linewidths=0.5,
        alpha=0.7
    )
    ax.add_collection(collection)
    
    # Draw bounding square
    if show_bounds:
        min_x, min_y, max_x, max_y = compute_bounding_box(all_vertices)
        width = max_x - min_x
        height = max_y - min_y
        side = max(width, height)
        
        # Center the square
        cx = (min_x + max_x) / 2
        cy = (min_y + max_y) / 2
        
        square = plt.Rectangle(
            (cx - side/2, cy - side/2),
            side, side,
            fill=False,
            edgecolor='blue',
            linestyle='--',
            linewidth=2
        )
        ax.add_patch(square)
    
    # Set axis limits with padding
    min_x, min_y, max_x, max_y = compute_bounding_box(all_vertices)
    padding = 0.2
    ax.set_xlim(min_x - padding, max_x + padding)
    ax.set_ylim(min_y - padding, max_y + padding)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Title
    if title is None:
        title = f"n={n} Trees"
    
    if show_score:
        score = compute_score(all_vertices, n)
        s = np.sqrt(score * n)
        title += f" | Score: {score:.6f} (s={s:.4f})"
        
        if colliding_trees:
            title += f" | ⚠️ {len(colliding_trees)} colliding"
    
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    return ax


def plot_solution(
    solution: np.ndarray,
    save_path: Optional[str] = None,
    **kwargs
):
    """
    Plot solution and optionally save to file.
    
    Args:
        solution: (n, 3) array
        save_path: Path to save figure (None = don't save)
        **kwargs: Passed to plot_trees
    """
    fig, ax = plt.subplots(1, 1, figsize=kwargs.pop('figsize', (10, 10)))
    plot_trees(solution, ax=ax, **kwargs)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    plt.show()


def plot_comparison(
    solutions: List[np.ndarray],
    titles: List[str] = None,
    figsize: Tuple[int, int] = (16, 8)
):
    """
    Plot multiple solutions side by side for comparison.
    
    Args:
        solutions: List of solution arrays
        titles: Titles for each solution
        figsize: Figure size
    """
    n_solutions = len(solutions)
    fig, axes = plt.subplots(1, n_solutions, figsize=figsize)
    
    if n_solutions == 1:
        axes = [axes]
    
    if titles is None:
        titles = [f"Solution {i+1}" for i in range(n_solutions)]
    
    for ax, solution, title in zip(axes, solutions, titles):
        plot_trees(solution, ax=ax, title=title)
    
    plt.tight_layout()
    plt.show()


def animate_optimization(
    history: List[np.ndarray],
    interval: int = 100,
    save_path: Optional[str] = None
):
    """
    Animate optimization progress.
    
    Args:
        history: List of solution arrays at each step
        interval: Milliseconds between frames
        save_path: Path to save animation (mp4/gif)
    """
    from matplotlib.animation import FuncAnimation
    
    if not history:
        print("No history to animate")
        return
    
    n = len(history[0])
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    def update(frame):
        ax.clear()
        plot_trees(
            history[frame],
            ax=ax,
            title=f"Step {frame}/{len(history)-1}"
        )
        return []
    
    anim = FuncAnimation(
        fig,
        update,
        frames=len(history),
        interval=interval,
        blit=False
    )
    
    if save_path:
        anim.save(save_path, writer='pillow' if save_path.endswith('.gif') else 'ffmpeg')
        print(f"Saved animation to {save_path}")
    
    plt.show()


def plot_score_history(
    scores: List[float],
    title: str = "Optimization Progress",
    save_path: Optional[str] = None
):
    """
    Plot score over optimization iterations.
    
    Args:
        scores: List of scores at each checkpoint
        title: Plot title
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(scores, 'b-', linewidth=1, alpha=0.7)
    ax.fill_between(range(len(scores)), scores, alpha=0.2)
    
    # Mark best
    best_idx = np.argmin(scores)
    best_score = scores[best_idx]
    ax.scatter([best_idx], [best_score], color='red', s=100, zorder=5, label=f'Best: {best_score:.6f}')
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Score (s²/n)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    
    plt.show()


def plot_all_scores(
    scores_by_n: dict,
    title: str = "Scores by Configuration",
    save_path: Optional[str] = None
):
    """
    Plot scores for all n values.
    
    Args:
        scores_by_n: Dict mapping n -> score
        title: Plot title
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ns = sorted(scores_by_n.keys())
    scores = [scores_by_n[n] for n in ns]
    
    ax.bar(ns, scores, color='green', alpha=0.7, edgecolor='darkgreen')
    
    # Highlight best and worst
    best_n = ns[np.argmin(scores)]
    worst_n = ns[np.argmax(scores)]
    
    ax.axhline(np.mean(scores), color='red', linestyle='--', label=f'Mean: {np.mean(scores):.4f}')
    
    ax.set_xlabel('n (trees)')
    ax.set_ylabel('Score (s²/n)')
    ax.set_title(f"{title} | Total: {sum(scores):.4f}")
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    
    plt.show()


# Quick test
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Plot base tree shape
    print("Plotting base tree shape...")
    fig, ax = plt.subplots(figsize=(6, 8))
    plot_tree_shape(ax)
    plt.show()
    
    # Create simple test solution
    print("\nPlotting test solution...")
    solution = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 45.0],
        [0.0, 1.2, 90.0],
        [1.0, 1.2, 135.0],
    ])
    plot_solution(solution, title="Test Configuration")
