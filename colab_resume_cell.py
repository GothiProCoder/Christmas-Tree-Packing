# =============================================================================
# 🎄 SANTA 2025 - COLAB RESUME CELL (COPY-PASTE THIS ENTIRE CELL) 🎄
# =============================================================================
# Run this cell when you come back to Colab after it crashed/disconnected.
# It will automatically resume from your last checkpoint saved in Google Drive.
# =============================================================================

# Step 1: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Step 2: Set environment variables (MUST be before other imports)
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMBA_NUM_THREADS'] = '1'
os.environ['NUMBA_THREADING_LAYER'] = 'workqueue'

# Step 3: Setup paths
DRIVE_PROJECT_DIR = '/content/drive/MyDrive/SANTA_2025'
REPO_URL = 'YOUR_GITHUB_REPO_URL_HERE'  # <-- CHANGE THIS!
LOCAL_DIR = '/content/SANTA'

# Step 4: Clone or pull latest code
import shutil
if os.path.exists(LOCAL_DIR):
    os.chdir(LOCAL_DIR)
    !git pull
else:
    !git clone {REPO_URL} {LOCAL_DIR}
    os.chdir(LOCAL_DIR)

# Step 5: Install dependencies (silent)
!pip install -q numpy numba scipy shapely cma tqdm matplotlib

# Step 6: Setup symlinks to Drive (for persistent checkpoints)
os.makedirs(f'{DRIVE_PROJECT_DIR}/checkpoints', exist_ok=True)
os.makedirs(f'{DRIVE_PROJECT_DIR}/results', exist_ok=True)

if os.path.exists('checkpoints') and not os.path.islink('checkpoints'):
    shutil.rmtree('checkpoints')
if os.path.exists('results') and not os.path.islink('results'):
    shutil.rmtree('results')
if not os.path.exists('checkpoints'):
    os.symlink(f'{DRIVE_PROJECT_DIR}/checkpoints', 'checkpoints')
if not os.path.exists('results'):
    os.symlink(f'{DRIVE_PROJECT_DIR}/results', 'results')

# Step 7: Check for existing checkpoint
checkpoint_path = f'{DRIVE_PROJECT_DIR}/checkpoints/latest.pkl'
if os.path.exists(checkpoint_path):
    import pickle
    with open(checkpoint_path, 'rb') as f:
        results = pickle.load(f)
    print(f"✅ Found checkpoint with {len(results)} completed configurations!")
    print(f"   Remaining: {200 - len(results)} configs")
    print(f"\n🚀 RESUMING from checkpoint...")
    
    # Resume!
    !python run_colab.py --full --workers 24 --resume checkpoints/latest.pkl
else:
    print("⚠️ No checkpoint found. Starting fresh...")
    !python run_colab.py --full --workers 24
