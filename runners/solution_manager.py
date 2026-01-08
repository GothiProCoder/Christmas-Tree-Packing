"""
Solution Manager - Track, Compare, Merge, and Manage Solutions

This is our STRATEGIC COMMAND for solution management:
1. Track best solutions found across multiple runs
2. Merge solutions from different optimization sessions
3. Compare solutions and identify improvement opportunities
4. Maintain solution history and statistics
5. Export/import solutions for sharing

Key insight: We'll run optimization MULTIPLE TIMES with different
strategies. The Solution Manager ensures we always keep the BEST
solution for each n, no matter which run produced it.

Features:
- Best solution tracking per n
- Solution merging (keep best from each source)
- Score analysis and statistics
- Improvement tracking over time
- Worst configuration identification
- Solution validation
- Export/import capabilities
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import pickle
import json
import os
import time
from datetime import datetime
import shutil


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class SolutionEntry:
    """Entry for a single n-tree solution."""
    n: int
    solution: np.ndarray  # (n, 3) array
    score: float
    method: str
    timestamp: str
    source: str = "unknown"
    iterations: int = 0
    validated: bool = False
    
    def to_dict(self) -> Dict:
        return {
            'n': self.n,
            'solution': self.solution.tolist(),
            'score': self.score,
            'method': self.method,
            'timestamp': self.timestamp,
            'source': self.source,
            'iterations': self.iterations,
            'validated': self.validated,
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'SolutionEntry':
        return cls(
            n=d['n'],
            solution=np.array(d['solution']),
            score=d['score'],
            method=d['method'],
            timestamp=d.get('timestamp', ''),
            source=d.get('source', 'unknown'),
            iterations=d.get('iterations', 0),
            validated=d.get('validated', False),
        )


@dataclass
class SolutionManagerConfig:
    """Configuration for solution manager."""
    storage_dir: str = "solutions"
    backup_dir: str = "solutions/backups"
    max_backups: int = 10
    auto_save: bool = True
    validate_on_add: bool = True


# =============================================================================
# SOLUTION MANAGER
# =============================================================================

class SolutionManager:
    """
    Manages solutions across multiple optimization runs.
    
    This is the SINGLE SOURCE OF TRUTH for best solutions.
    
    Example:
        manager = SolutionManager()
        
        # Add solutions from different runs
        manager.add_from_results(run1_results, source="run1")
        manager.add_from_results(run2_results, source="run2")
        
        # Automatically keeps the best for each n
        print(f"Total score: {manager.total_score}")
        
        # Export for submission
        manager.export_solutions("best_solutions.pkl")
    """
    
    def __init__(self, config: SolutionManagerConfig = None):
        self.config = config or SolutionManagerConfig()
        
        # Main storage: n -> SolutionEntry
        self.solutions: Dict[int, SolutionEntry] = {}
        
        # History tracking
        self.history: List[Dict] = []  # Log of all additions
        self.score_history: Dict[int, List[Tuple[str, float]]] = {}  # n -> [(timestamp, score)]
        
        # Statistics
        self.stats = {
            'total_additions': 0,
            'improvements': 0,
            'last_update': None,
        }
        
        # Ensure directories exist
        os.makedirs(self.config.storage_dir, exist_ok=True)
        os.makedirs(self.config.backup_dir, exist_ok=True)
    
    # =========================================================================
    # CORE OPERATIONS
    # =========================================================================
    
    def add_solution(
        self,
        n: int,
        solution: np.ndarray,
        score: float,
        method: str = "unknown",
        source: str = "unknown",
        iterations: int = 0,
        force: bool = False
    ) -> bool:
        """
        Add a solution. Keeps it only if it's better than existing.
        
        Args:
            n: Number of trees
            solution: (n, 3) array of [x, y, angle_deg]
            score: s²/n score
            method: Optimization method used
            source: Identifier for the run/session
            iterations: Number of iterations used
            force: If True, replace even if worse
            
        Returns:
            True if solution was added/updated, False otherwise
        """
        timestamp = datetime.now().isoformat()
        
        # Validate if configured
        if self.config.validate_on_add:
            is_valid, issues = self._validate_solution(n, solution)
            if not is_valid:
                print(f"  ⚠️ Solution for n={n} failed validation: {issues[:2]}")
                return False
        
        # Check if we should add
        should_add = force
        is_improvement = False
        
        if n not in self.solutions:
            should_add = True
            is_improvement = True
        elif score < self.solutions[n].score:
            should_add = True
            is_improvement = True
            improvement = self.solutions[n].score - score
            
            # Log improvement
            self.history.append({
                'type': 'improvement',
                'n': n,
                'old_score': self.solutions[n].score,
                'new_score': score,
                'improvement': improvement,
                'method': method,
                'source': source,
                'timestamp': timestamp,
            })
        
        if should_add:
            entry = SolutionEntry(
                n=n,
                solution=solution.copy(),
                score=score,
                method=method,
                timestamp=timestamp,
                source=source,
                iterations=iterations,
                validated=self.config.validate_on_add,
            )
            
            self.solutions[n] = entry
            
            # Track score history
            if n not in self.score_history:
                self.score_history[n] = []
            self.score_history[n].append((timestamp, score))
            
            # Update stats
            self.stats['total_additions'] += 1
            if is_improvement:
                self.stats['improvements'] += 1
            self.stats['last_update'] = timestamp
            
            # Auto-save
            if self.config.auto_save:
                self._auto_save()
            
            return True
        
        return False
    
    def add_from_results(
        self,
        results: Dict,
        source: str = "unknown"
    ) -> Tuple[int, int]:
        """
        Add solutions from an optimization results dictionary.
        
        Args:
            results: Dict mapping n -> OptimizationResult or solution array
            source: Identifier for this batch
            
        Returns:
            (n_added, n_improved)
        """
        n_added = 0
        n_improved = 0
        
        for n, result in results.items():
            # Handle different result types
            if hasattr(result, 'solution'):
                solution = result.solution
                score = result.score
                method = getattr(result, 'method', 'unknown')
                iterations = getattr(result, 'iterations', 0)
            elif isinstance(result, np.ndarray):
                solution = result
                score = self._compute_score(solution)
                method = 'unknown'
                iterations = 0
            else:
                continue
            
            was_new = n not in self.solutions
            added = self.add_solution(
                n=n,
                solution=solution,
                score=score,
                method=method,
                source=source,
                iterations=iterations
            )
            
            if added:
                n_added += 1
                if not was_new:
                    n_improved += 1
        
        return n_added, n_improved
    
    def get_solution(self, n: int) -> Optional[np.ndarray]:
        """Get the best solution for n."""
        if n in self.solutions:
            return self.solutions[n].solution.copy()
        return None
    
    def get_score(self, n: int) -> Optional[float]:
        """Get the best score for n."""
        if n in self.solutions:
            return self.solutions[n].score
        return None
    
    def get_entry(self, n: int) -> Optional[SolutionEntry]:
        """Get the full solution entry for n."""
        return self.solutions.get(n)
    
    # =========================================================================
    # ANALYSIS
    # =========================================================================
    
    @property
    def total_score(self) -> float:
        """Total score across all solutions."""
        return sum(entry.score for entry in self.solutions.values())
    
    @property
    def n_complete(self) -> int:
        """Number of configurations with solutions."""
        return len(self.solutions)
    
    @property
    def n_missing(self) -> List[int]:
        """List of n values without solutions."""
        return [n for n in range(1, 201) if n not in self.solutions]
    
    def get_worst_configurations(self, k: int = 10) -> List[Tuple[int, float, str]]:
        """
        Get the k worst-scoring configurations.
        
        Returns:
            List of (n, score, method) tuples
        """
        sorted_entries = sorted(
            self.solutions.values(),
            key=lambda e: e.score,
            reverse=True
        )
        
        return [
            (e.n, e.score, e.method)
            for e in sorted_entries[:k]
        ]
    
    def get_best_configurations(self, k: int = 10) -> List[Tuple[int, float, str]]:
        """Get the k best-scoring configurations."""
        sorted_entries = sorted(
            self.solutions.values(),
            key=lambda e: e.score
        )
        
        return [
            (e.n, e.score, e.method)
            for e in sorted_entries[:k]
        ]
    
    def get_score_by_range(self) -> Dict[str, float]:
        """Get total scores broken down by n ranges."""
        ranges = {
            'n=1-12': (1, 12),
            'n=13-40': (13, 40),
            'n=41-100': (41, 100),
            'n=101-200': (101, 200),
        }
        
        scores = {}
        for name, (lo, hi) in ranges.items():
            scores[name] = sum(
                self.solutions[n].score
                for n in range(lo, hi + 1)
                if n in self.solutions
            )
        
        return scores
    
    def get_method_stats(self) -> Dict[str, Dict]:
        """Get statistics by optimization method."""
        method_stats = {}
        
        for entry in self.solutions.values():
            method = entry.method
            if method not in method_stats:
                method_stats[method] = {
                    'count': 0,
                    'total_score': 0,
                    'scores': [],
                }
            
            method_stats[method]['count'] += 1
            method_stats[method]['total_score'] += entry.score
            method_stats[method]['scores'].append(entry.score)
        
        # Compute averages
        for method, stats in method_stats.items():
            stats['avg_score'] = stats['total_score'] / stats['count']
            del stats['scores']  # Remove list to keep it clean
        
        return method_stats
    
    def print_summary(self):
        """Print a comprehensive summary."""
        print(f"\n{'='*60}")
        print("SOLUTION MANAGER SUMMARY")
        print(f"{'='*60}")
        
        print(f"\n  Total Score: {self.total_score:.6f}")
        print(f"  Configurations: {self.n_complete}/200")
        
        if self.n_missing:
            print(f"  Missing: {len(self.n_missing)} ({self.n_missing[:5]}...)")
        
        print(f"\n  Score by Range:")
        for name, score in self.get_score_by_range().items():
            print(f"    {name}: {score:.4f}")
        
        print(f"\n  Method Distribution:")
        for method, stats in self.get_method_stats().items():
            print(f"    {method}: {stats['count']} configs, "
                  f"avg={stats['avg_score']:.4f}")
        
        print(f"\n  Worst Configurations:")
        for n, score, method in self.get_worst_configurations(5):
            print(f"    n={n:3d}: {score:.6f} ({method})")
        
        print(f"\n  Statistics:")
        print(f"    Total additions: {self.stats['total_additions']}")
        print(f"    Improvements: {self.stats['improvements']}")
        print(f"    Last update: {self.stats['last_update']}")
        
        print(f"{'='*60}")
    
    # =========================================================================
    # MERGING
    # =========================================================================
    
    def merge_from(
        self,
        other: 'SolutionManager',
        source: str = "merged"
    ) -> Tuple[int, int]:
        """
        Merge solutions from another SolutionManager.
        
        Keeps the best solution for each n from either source.
        
        Returns:
            (n_added, n_improved)
        """
        n_added = 0
        n_improved = 0
        
        for n, entry in other.solutions.items():
            was_new = n not in self.solutions
            added = self.add_solution(
                n=n,
                solution=entry.solution,
                score=entry.score,
                method=entry.method,
                source=f"{source}:{entry.source}",
                iterations=entry.iterations
            )
            
            if added:
                n_added += 1
                if not was_new:
                    n_improved += 1
        
        return n_added, n_improved
    
    def merge_from_file(self, filepath: str) -> Tuple[int, int]:
        """Merge solutions from a saved file."""
        other = SolutionManager()
        other.load(filepath)
        
        source = os.path.basename(filepath)
        return self.merge_from(other, source=source)
    
    # =========================================================================
    # PERSISTENCE
    # =========================================================================
    
    def save(self, filepath: str = None):
        """Save all solutions to file."""
        if filepath is None:
            filepath = os.path.join(self.config.storage_dir, "solutions.pkl")
        
        data = {
            'solutions': {n: e.to_dict() for n, e in self.solutions.items()},
            'history': self.history,
            'score_history': {
                n: hist for n, hist in self.score_history.items()
            },
            'stats': self.stats,
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"  Saved {len(self.solutions)} solutions to {filepath}")
    
    def load(self, filepath: str = None):
        """Load solutions from file."""
        if filepath is None:
            filepath = os.path.join(self.config.storage_dir, "solutions.pkl")
        
        if not os.path.exists(filepath):
            print(f"  No saved solutions found at {filepath}")
            return
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.solutions = {
            int(n): SolutionEntry.from_dict(e)
            for n, e in data['solutions'].items()
        }
        self.history = data.get('history', [])
        self.score_history = data.get('score_history', {})
        self.stats = data.get('stats', self.stats)
        
        print(f"  Loaded {len(self.solutions)} solutions from {filepath}")
    
    def export_for_submission(self, filepath: str) -> Dict[int, np.ndarray]:
        """Export solutions in format suitable for submission."""
        solutions = {n: entry.solution for n, entry in self.solutions.items()}
        
        with open(filepath, 'wb') as f:
            pickle.dump(solutions, f)
        
        return solutions
    
    def _auto_save(self):
        """Automatic save to default location."""
        filepath = os.path.join(self.config.storage_dir, "solutions_auto.pkl")
        self.save(filepath)
    
    def create_backup(self) -> str:
        """Create a backup of current solutions."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(
            self.config.backup_dir,
            f"backup_{timestamp}.pkl"
        )
        
        self.save(backup_path)
        
        # Clean old backups
        self._cleanup_old_backups()
        
        return backup_path
    
    def _cleanup_old_backups(self):
        """Remove old backups beyond max_backups."""
        backups = sorted([
            f for f in os.listdir(self.config.backup_dir)
            if f.startswith('backup_') and f.endswith('.pkl')
        ])
        
        while len(backups) > self.config.max_backups:
            oldest = backups.pop(0)
            os.remove(os.path.join(self.config.backup_dir, oldest))
    
    # =========================================================================
    # VALIDATION
    # =========================================================================
    
    def _validate_solution(
        self,
        n: int,
        solution: np.ndarray
    ) -> Tuple[bool, List[str]]:
        """Validate a solution."""
        issues = []
        
        # Check dimensions
        if solution.shape != (n, 3):
            issues.append(f"Wrong shape: expected ({n}, 3), got {solution.shape}")
        
        # Check bounds
        for i in range(n):
            x, y, angle = solution[i]
            if abs(x) > 100:
                issues.append(f"Tree {i}: x={x:.2f} out of bounds")
            if abs(y) > 100:
                issues.append(f"Tree {i}: y={y:.2f} out of bounds")
        
        # Check collisions
        from core.collision import validate_solution
        valid, collisions = validate_solution(solution)
        if not valid:
            issues.append(f"{len(collisions)} collision(s) detected")
        
        return len(issues) == 0, issues
    
    def validate_all(self) -> Dict[int, List[str]]:
        """Validate all solutions."""
        issues = {}
        
        for n, entry in self.solutions.items():
            is_valid, problems = self._validate_solution(n, entry.solution)
            if not is_valid:
                issues[n] = problems
                entry.validated = False
            else:
                entry.validated = True
        
        return issues
    
    def _compute_score(self, solution: np.ndarray) -> float:
        """Compute score for a solution."""
        from core.tree_polygon import transform_trees_batch
        from core.bounding_box import compute_score
        
        n = len(solution)
        positions = solution[:, :2]
        angles = np.radians(solution[:, 2])
        
        all_verts = transform_trees_batch(positions, angles)
        return compute_score(all_verts, n)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_manager() -> SolutionManager:
    """Create a new solution manager with default config."""
    return SolutionManager()


def load_best_solutions(filepath: str = "solutions/solutions.pkl") -> Dict[int, np.ndarray]:
    """Load best solutions from file."""
    manager = SolutionManager()
    manager.load(filepath)
    return {n: entry.solution for n, entry in manager.solutions.items()}


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    print("Testing Solution Manager")
    print("=" * 60)
    
    # Create manager
    manager = SolutionManager()
    
    # Add some dummy solutions
    print("\nAdding test solutions...")
    
    for n in [1, 5, 10, 20, 50]:
        # Create dummy solution
        sqrt_n = int(np.ceil(np.sqrt(n)))
        solution = np.zeros((n, 3))
        idx = 0
        for i in range(sqrt_n):
            for j in range(sqrt_n):
                if idx >= n:
                    break
                solution[idx] = [i * 1.5 - sqrt_n * 0.75, j * 1.5 - sqrt_n * 0.75, 0]
                idx += 1
        
        # Compute score
        from core.tree_polygon import warmup as warmup_tree
        warmup_tree()
        score = manager._compute_score(solution)
        
        added = manager.add_solution(
            n=n,
            solution=solution,
            score=score,
            method='test',
            source='test_run'
        )
        
        print(f"  n={n:3d}: score={score:.6f}, added={added}")
    
    # Print summary
    manager.print_summary()
    
    # Test improvement
    print("\nTesting improvement tracking...")
    
    # Add a better solution for n=10
    n = 10
    solution = manager.get_solution(n)
    solution[:, 0] *= 0.9  # Compress slightly
    solution[:, 1] *= 0.9
    
    new_score = manager._compute_score(solution)
    added = manager.add_solution(
        n=n,
        solution=solution,
        score=new_score,
        method='test_improved',
        source='test_run_2'
    )
    
    print(f"  n={n}: new_score={new_score:.6f}, improved={added}")
    
    # Final summary
    print(f"\n✅ Test complete!")
    print(f"   Total score: {manager.total_score:.6f}")
