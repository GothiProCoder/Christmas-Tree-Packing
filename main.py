#!/usr/bin/env python3
"""
🎄 SANTA 2025 - CHRISTMAS TREE PACKING OPTIMIZER 🎄

                    ★
                   /|\
                  /_|_\
                   /|\
                  / | \
                 /  |  \
                /___|___\
                   |||
                   |||
              ═════════════

This is the MAIN ENTRY POINT for the competition solution.

Usage:
    python main.py --full                    # Run full optimization (all 200 configs)
    python main.py --quick                   # Quick test run
    python main.py --single 50               # Optimize single n
    python main.py --range 1 50              # Optimize range of n
    python main.py --resume checkpoint.pkl   # Resume from checkpoint
    python main.py --submit                  # Generate submission file
    python main.py --analyze                 # Analyze current solutions
    python main.py --visualize 10            # Visualize solution for n

Strategy:
    n=1-2:     Exhaustive/Grid search (trivial cases)
    n=3-12:    CMA-ES + Basin Hopping (heavy artillery)
    n=13-40:   Adaptive SA (2M+ iterations)
    n=41-100:  Hybrid Lattice + SA
    n=101-200: Lattice + Edge optimization

Target Score: < 69.0 (top-tier)
"""

# =============================================================================
# FIX FOR LINUX/COLAB: Prevent OpenMP + fork() crash
# MUST be set BEFORE importing numpy/numba
# =============================================================================
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1' 
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMBA_NUM_THREADS'] = '1'

import argparse
import sys
import time
from datetime import datetime

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


# =============================================================================
# BANNER
# =============================================================================

BANNER = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   🎄  SANTA 2025 - CHRISTMAS TREE PACKING OPTIMIZER  🎄                     ║
║                                                                              ║
║        ★                                                                     ║
║       /|\\                Target Score: < 69.0                               ║
║      /_|_\\               Configurations: 200                                ║
║       /|\\                Strategy: Adaptive Multi-Method                    ║
║      / | \\                                                                   ║
║     /__|__\\                                                                  ║
║        ||                                                                    ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""


# =============================================================================
# WARMUP
# =============================================================================

def warmup_jit():
    """Warm up JIT compilation for all modules."""
    print("\n🔥 Warming up JIT compilation...")
    
    from core.tree_polygon import warmup as warmup_tree
    warmup_tree()
    
    try:
        from core.bounding_box import warmup as warmup_bbox
        warmup_bbox()
    except ImportError:
        pass
    
    print("   ✅ JIT warmup complete!")


# =============================================================================
# COMMANDS
# =============================================================================

def cmd_full(args):
    """Run full optimization on all 200 configurations."""
    from runners.parallel_runner import ParallelOptimizer, RunnerConfig
    from runners.solution_manager import SolutionManager
    
    print(BANNER)
    warmup_jit()
    
    # Configure
    config = RunnerConfig(
        n_workers=args.workers,
        cmaes_evaluations=args.cmaes_evals,
        sa_iterations=args.sa_iters,
        hybrid_sa_iterations=args.hybrid_iters,
        lattice_refinement_iters=args.lattice_iters,
        checkpoint_dir=args.checkpoint_dir,
        results_dir=args.results_dir,
        verbose=True,
        progress_bar=True,
    )
    
    print(f"\n📋 Configuration:")
    print(f"   Workers: {config.n_workers}")
    print(f"   CMA-ES evals: {config.cmaes_evaluations:,}")
    print(f"   SA iterations: {config.sa_iterations:,}")
    print(f"   Hybrid SA iters: {config.hybrid_sa_iterations:,}")
    print(f"   Lattice refinement: {config.lattice_refinement_iters:,}")
    
    # Run
    optimizer = ParallelOptimizer(config)
    
    start_time = time.time()
    
    if args.resume:
        results = optimizer.optimize_all(resume_from=args.resume)
    else:
        results = optimizer.optimize_all()
    
    elapsed = time.time() - start_time
    
    # Save to solution manager
    manager = SolutionManager()
    manager.add_from_results(results, source="main_full_run")
    manager.save()
    manager.print_summary()
    
    # Generate submission
    if args.submit:
        submission_path = optimizer.generate_submission()
        print(f"\n📄 Submission saved: {submission_path}")
    
    print(f"\n⏱️  Total time: {elapsed/3600:.2f} hours")
    
    return results


def cmd_quick(args):
    """Quick test run with reduced iterations."""
    from runners.parallel_runner import quick_test
    
    print(BANNER)
    warmup_jit()
    
    test_ns = [1, 2, 5, 10, 20, 50, 100, 150, 200]
    
    print(f"\n🚀 Quick test with n = {test_ns}")
    print("   (Using reduced iterations for speed)")
    
    results = quick_test(test_ns, verbose=True)
    
    total = sum(r.score for r in results.values())
    print(f"\n📊 Subset total score: {total:.6f}")
    
    # Estimate full score
    estimated_full = total * (200 / len(test_ns))
    print(f"   Estimated full score: ~{estimated_full:.1f}")
    
    return results


def cmd_single(args):
    """Optimize a single configuration."""
    from runners.parallel_runner import ParallelOptimizer, RunnerConfig, get_strategy
    from runners.solution_manager import SolutionManager
    
    print(BANNER)
    warmup_jit()
    
    n = args.n
    
    config = RunnerConfig(
        cmaes_evaluations=args.cmaes_evals,
        sa_iterations=args.sa_iters,
        verbose=True,
    )
    
    strategy = get_strategy(n, config)
    
    print(f"\n🎯 Optimizing n={n}")
    print(f"   Strategy: {strategy}")
    
    optimizer = ParallelOptimizer(config)
    result = optimizer.optimize_single(n)
    
    print(f"\n✅ Result:")
    print(f"   Score: {result.score:.8f}")
    print(f"   Method: {result.method}")
    print(f"   Time: {result.time_seconds:.1f}s")
    
    # Save
    manager = SolutionManager()
    try:
        manager.load()
    except:
        pass
    
    manager.add_solution(
        n=n,
        solution=result.solution,
        score=result.score,
        method=result.method,
        source="main_single"
    )
    manager.save()
    
    return result


def cmd_range(args):
    """Optimize a range of configurations."""
    from runners.parallel_runner import ParallelOptimizer, RunnerConfig
    from runners.solution_manager import SolutionManager
    
    print(BANNER)
    warmup_jit()
    
    n_min, n_max = args.n_min, args.n_max
    
    config = RunnerConfig(
        n_workers=args.workers,
        cmaes_evaluations=args.cmaes_evals,
        sa_iterations=args.sa_iters,
        hybrid_sa_iterations=args.hybrid_iters,
        lattice_refinement_iters=args.lattice_iters,
        verbose=True,
        progress_bar=True,
    )
    
    print(f"\n🎯 Optimizing range n={n_min} to n={n_max}")
    
    optimizer = ParallelOptimizer(config)
    results = optimizer.optimize_all(n_range=(n_min, n_max))
    
    # Save
    manager = SolutionManager()
    try:
        manager.load()
    except:
        pass
    
    manager.add_from_results(results, source=f"main_range_{n_min}_{n_max}")
    manager.save()
    manager.print_summary()
    
    return results


def cmd_submit(args):
    """Generate submission file from saved solutions."""
    from runners.solution_manager import SolutionManager
    from utils.submission import generate_submission
    
    print(BANNER)
    
    manager = SolutionManager()
    manager.load()
    
    if manager.n_complete < 200:
        print(f"\n⚠️  Warning: Only {manager.n_complete}/200 configurations complete!")
        print(f"   Missing: {manager.n_missing[:10]}...")
        
        if not args.force:
            print("\n   Use --force to generate partial submission")
            return
    
    solutions = {n: entry.solution for n, entry in manager.solutions.items()}
    
    output_path = args.output or "submission.csv"
    success = generate_submission(solutions, output_path, validate=True, verbose=True)
    
    if success:
        print(f"\n✅ Submission saved: {output_path}")
        print(f"   Total score: {manager.total_score:.6f}")
    else:
        print("\n❌ Submission generation failed!")


def cmd_analyze(args):
    """Analyze current solutions."""
    from runners.solution_manager import SolutionManager
    
    print(BANNER)
    
    manager = SolutionManager()
    manager.load()
    
    manager.print_summary()
    
    print("\n📈 Improvement Opportunities:")
    print("\n   Worst 10 configurations to target:")
    for n, score, method in manager.get_worst_configurations(10):
        print(f"     n={n:3d}: {score:.6f} ({method})")
    
    print("\n   Score comparison by range:")
    for name, score in manager.get_score_by_range().items():
        # Estimate "good" score for comparison
        if name == 'n=1-12':
            target = 2.5
        elif name == 'n=13-40':
            target = 10.0
        elif name == 'n=41-100':
            target = 25.0
        else:
            target = 30.0
        
        status = "✅" if score < target else "⚠️"
        print(f"     {name}: {score:.4f} (target: ~{target:.1f}) {status}")


def cmd_visualize(args):
    """Visualize a solution."""
    from runners.solution_manager import SolutionManager
    from utils.visualization import plot_solution
    
    print(BANNER)
    
    manager = SolutionManager()
    manager.load()
    
    n = args.n
    
    if n not in manager.solutions:
        print(f"\n❌ No solution found for n={n}")
        return
    
    entry = manager.solutions[n]
    
    print(f"\n🎨 Visualizing n={n}")
    print(f"   Score: {entry.score:.6f}")
    print(f"   Method: {entry.method}")
    
    save_path = args.save
    plot_solution(entry.solution, save_path=save_path, title=f"n={n}")


def cmd_improve(args):
    """Run additional optimization on worst configurations."""
    from runners.parallel_runner import ParallelOptimizer, RunnerConfig
    from runners.solution_manager import SolutionManager
    
    print(BANNER)
    warmup_jit()
    
    manager = SolutionManager()
    manager.load()
    
    # Find worst k configurations
    k = args.k or 20
    worst = manager.get_worst_configurations(k)
    
    print(f"\n🎯 Targeting {k} worst configurations for improvement:")
    for n, score, method in worst[:5]:
        print(f"   n={n:3d}: {score:.6f} ({method})")
    print("   ...")
    
    # Run with increased iterations
    config = RunnerConfig(
        n_workers=args.workers,
        cmaes_evaluations=args.cmaes_evals * 2,
        sa_iterations=args.sa_iters * 2,
        verbose=True,
        progress_bar=True,
    )
    
    optimizer = ParallelOptimizer(config)
    
    # Only optimize worst configs
    n_values = [n for n, _, _ in worst]
    n_min, n_max = min(n_values), max(n_values)
    
    # Load existing as starting point
    existing = {n: entry.solution for n, entry in manager.solutions.items()}
    
    # Run optimization
    results = optimizer.optimize_all(n_range=(n_min, n_max))
    
    # Merge improvements
    n_improved = 0
    for n, result in results.items():
        old_score = manager.get_score(n) or float('inf')
        if result.score < old_score:
            manager.add_solution(
                n=n,
                solution=result.solution,
                score=result.score,
                method=result.method,
                source="improvement_run"
            )
            n_improved += 1
    
    manager.save()
    
    print(f"\n✅ Improved {n_improved} configurations!")
    print(f"   New total score: {manager.total_score:.6f}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="🎄 SANTA 2025 Christmas Tree Packing Optimizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py --full                    # Full optimization
    python main.py --quick                   # Quick test
    python main.py --single 50               # Single config
    python main.py --range 1 50              # Range of configs
    python main.py --submit                  # Generate submission
    python main.py --analyze                 # Analyze solutions
    python main.py --improve --k 20          # Improve worst 20
        """
    )
    
    # Commands
    cmd_group = parser.add_mutually_exclusive_group(required=True)
    cmd_group.add_argument('--full', action='store_true', help='Run full optimization')
    cmd_group.add_argument('--quick', action='store_true', help='Quick test run')
    cmd_group.add_argument('--single', type=int, metavar='N', dest='n', help='Optimize single n')
    cmd_group.add_argument('--range', nargs=2, type=int, metavar=('MIN', 'MAX'), help='Optimize range')
    cmd_group.add_argument('--submit', action='store_true', help='Generate submission')
    cmd_group.add_argument('--analyze', action='store_true', help='Analyze solutions')
    cmd_group.add_argument('--visualize', type=int, metavar='N', dest='viz_n', help='Visualize solution')
    cmd_group.add_argument('--improve', action='store_true', help='Improve worst configs')
    
    # Options
    parser.add_argument('--workers', type=int, default=None, help='Number of parallel workers')
    parser.add_argument('--resume', type=str, metavar='PATH', help='Resume from checkpoint')
    parser.add_argument('--output', type=str, metavar='PATH', help='Output path for submission')
    parser.add_argument('--save', type=str, metavar='PATH', help='Save visualization to file')
    parser.add_argument('--force', action='store_true', help='Force operation')
    parser.add_argument('--k', type=int, default=20, help='Number of configs for --improve')
    
    # Iteration controls
    parser.add_argument('--cmaes-evals', type=int, default=100000, help='CMA-ES evaluations')
    parser.add_argument('--sa-iters', type=int, default=2000000, help='SA iterations')
    parser.add_argument('--hybrid-iters', type=int, default=500000, help='Hybrid SA iterations')
    parser.add_argument('--lattice-iters', type=int, default=200000, help='Lattice refinement iterations')
    
    # Directories
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--results-dir', type=str, default='results', help='Results directory')
    
    args = parser.parse_args()
    
    # Route to command
    try:
        if args.full:
            cmd_full(args)
        elif args.quick:
            cmd_quick(args)
        elif args.n is not None:
            cmd_single(args)
        elif args.range:
            args.n_min, args.n_max = args.range
            cmd_range(args)
        elif args.submit:
            cmd_submit(args)
        elif args.analyze:
            cmd_analyze(args)
        elif args.viz_n is not None:
            args.n = args.viz_n
            cmd_visualize(args)
        elif args.improve:
            cmd_improve(args)
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        print("   Progress has been checkpointed.")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
