#!/usr/bin/env python3
"""
🎄 SANTA 2025 - COLAB RUNNER (USE THIS FOR COLAB) 🎄

This script sets environment variables BEFORE importing anything,
which fixes the OpenMP + fork() crash on Linux/Colab.

Usage in Colab:
    !python run_colab.py --full --workers 24
    !python run_colab.py --quick
"""

# =============================================================================
# CRITICAL: Set these BEFORE any other imports!
# =============================================================================
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1' 
os.environ['NUMBA_NUM_THREADS'] = '1'
os.environ['NUMBA_THREADING_LAYER'] = 'workqueue'

# Now we can safely import and run main
import sys

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# Import main module (after env vars are set)
if __name__ == "__main__":
    # Pass all arguments to main
    from main import main
    main()
