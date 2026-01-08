"""
Submission Generator - Create valid competition submission files.

The submission format requires:
- id: NNN_idx format (e.g., 001_0, 002_1)
- x, y, deg: Values prefixed with 's' (e.g., s0.0, s45.123)
- No overlapping trees
- Coordinates within [-100, 100]
"""

import numpy as np
import csv
from typing import Dict, List, Tuple, Optional
from decimal import Decimal, getcontext
import os

# Set high precision for submission values
getcontext().prec = 50


def format_value(value: float, precision: int = 15) -> str:
    """
    Format a value for submission with 's' prefix.
    
    Args:
        value: Numeric value
        precision: Decimal places (15 recommended for competition)
        
    Returns:
        Formatted string like "s1.234567890123456"
    """
    # Use Decimal for precision
    d = Decimal(str(value))
    formatted = f"{d:.{precision}f}"
    return f"s{formatted}"


def validate_solution_for_submission(
    solution: np.ndarray,
    n: int
) -> Tuple[bool, List[str]]:
    """
    Validate a solution before submission.
    
    Args:
        solution: (n, 3) array of [x, y, angle_deg]
        n: Expected number of trees
        
    Returns:
        (is_valid, list_of_issues)
    """
    issues = []
    
    # Check dimensions
    if len(solution) != n:
        issues.append(f"Expected {n} trees, got {len(solution)}")
    
    # Check coordinate bounds [-100, 100]
    for i, (x, y, deg) in enumerate(solution):
        if abs(x) > 100:
            issues.append(f"Tree {i}: x={x:.4f} out of bounds [-100, 100]")
        if abs(y) > 100:
            issues.append(f"Tree {i}: y={y:.4f} out of bounds [-100, 100]")
    
    # Check for collisions
    from core.collision import validate_solution
    valid, collisions = validate_solution(solution)
    if not valid:
        for i, j in collisions[:5]:  # Only report first 5
            issues.append(f"Collision between trees {i} and {j}")
        if len(collisions) > 5:
            issues.append(f"... and {len(collisions) - 5} more collisions")
    
    return len(issues) == 0, issues


def generate_submission(
    solutions: Dict[int, np.ndarray],
    output_path: str = "submission.csv",
    validate: bool = True,
    verbose: bool = True
) -> bool:
    """
    Generate a complete submission file.
    
    Args:
        solutions: Dict mapping n -> solution array (n, 3)
        output_path: Path to output CSV file
        validate: Run validation checks
        verbose: Print progress
        
    Returns:
        True if successful, False if validation failed
    """
    # Check all n values are present
    missing = [n for n in range(1, 201) if n not in solutions]
    if missing:
        print(f"ERROR: Missing solutions for n = {missing}")
        return False
    
    # Validate all solutions
    if validate:
        if verbose:
            print("Validating solutions...")
        
        all_valid = True
        for n in range(1, 201):
            solution = solutions[n]
            is_valid, issues = validate_solution_for_submission(solution, n)
            
            if not is_valid:
                print(f"\n❌ n={n} INVALID:")
                for issue in issues:
                    print(f"   - {issue}")
                all_valid = False
        
        if not all_valid:
            print("\n⚠️ Validation failed. Fix issues before submitting.")
            return False
        
        if verbose:
            print("✅ All solutions valid!")
    
    # Generate CSV
    if verbose:
        print(f"Generating submission file: {output_path}")
    
    rows = []
    for n in range(1, 201):
        solution = solutions[n]
        for tree_idx in range(n):
            x, y, deg = solution[tree_idx]
            
            row = {
                'id': f"{n:03d}_{tree_idx}",
                'x': format_value(x),
                'y': format_value(y),
                'deg': format_value(deg)
            }
            rows.append(row)
    
    # Write CSV
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'x', 'y', 'deg'])
        writer.writeheader()
        writer.writerows(rows)
    
    # Compute total score
    total_score = 0.0
    from core.bounding_box import get_score
    for n in range(1, 201):
        total_score += get_score(solutions[n])
    
    if verbose:
        row_count = sum(n for n in range(1, 201))  # 1 + 2 + ... + 200 = 20100
        print(f"✅ Submission saved: {output_path}")
        print(f"   - Total rows: {row_count}")
        print(f"   - Total score: {total_score:.6f}")
    
    return True


def generate_submission_from_results(
    results: dict,
    output_path: str = "submission.csv"
) -> bool:
    """
    Generate submission from OptimizationResult objects.
    
    Args:
        results: Dict mapping n -> OptimizationResult
        output_path: Output path
        
    Returns:
        Success status
    """
    solutions = {}
    for n, result in results.items():
        if hasattr(result, 'solution'):
            solutions[n] = result.solution
        else:
            solutions[n] = result
    
    return generate_submission(solutions, output_path)


def parse_submission(filepath: str) -> Dict[int, np.ndarray]:
    """
    Parse a submission file back into solution arrays.
    
    Args:
        filepath: Path to submission CSV
        
    Returns:
        Dict mapping n -> solution array
    """
    solutions = {n: [] for n in range(1, 201)}
    
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Parse id
            n_str, idx_str = row['id'].split('_')
            n = int(n_str)
            idx = int(idx_str)
            
            # Parse values (remove 's' prefix)
            x = float(row['x'][1:])
            y = float(row['y'][1:])
            deg = float(row['deg'][1:])
            
            solutions[n].append([x, y, deg])
    
    # Convert to arrays
    for n in range(1, 201):
        solutions[n] = np.array(solutions[n])
    
    return solutions


def compute_submission_score(filepath: str) -> float:
    """
    Compute total score from a submission file.
    
    Args:
        filepath: Path to submission CSV
        
    Returns:
        Total score
    """
    solutions = parse_submission(filepath)
    
    from core.bounding_box import get_score
    
    total = 0.0
    for n in range(1, 201):
        total += get_score(solutions[n])
    
    return total


def save_solutions_pickle(solutions: dict, filepath: str):
    """Save solutions dict to pickle for checkpointing."""
    import pickle
    with open(filepath, 'wb') as f:
        pickle.dump(solutions, f)
    print(f"Saved solutions to {filepath}")


def load_solutions_pickle(filepath: str) -> dict:
    """Load solutions dict from pickle."""
    import pickle
    with open(filepath, 'rb') as f:
        solutions = pickle.load(f)
    print(f"Loaded solutions from {filepath}")
    return solutions


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Create dummy solutions for testing
    print("Creating test solutions...")
    
    solutions = {}
    for n in range(1, 11):  # Just test first 10
        # Simple grid placement
        sqrt_n = int(np.ceil(np.sqrt(n)))
        solution = np.zeros((n, 3))
        for i in range(n):
            solution[i, 0] = (i % sqrt_n) * 1.5
            solution[i, 1] = (i // sqrt_n) * 1.5
            solution[i, 2] = 0.0
        solutions[n] = solution
    
    # Test validation
    print("\nTesting validation...")
    for n in range(1, 6):
        is_valid, issues = validate_solution_for_submission(solutions[n], n)
        print(f"n={n}: {'✅' if is_valid else '❌'}")
    
    # Test formatting
    print("\nTest value formatting:")
    print(f"  0.0 -> {format_value(0.0)}")
    print(f"  3.14159 -> {format_value(3.14159)}")
    print(f"  -45.678 -> {format_value(-45.678)}")
