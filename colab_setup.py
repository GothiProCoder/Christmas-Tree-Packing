# =============================================================================
# 🎄 SANTA 2025 - COLAB SETUP WITH GOOGLE DRIVE PERSISTENCE 🎄
# =============================================================================
# Run this cell first in your Colab notebook to set everything up.
# Checkpoints will be saved to Google Drive for crash recovery.
# =============================================================================

# Step 1: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Step 2: Create project directory in Drive (first time only)
import os
import shutil

DRIVE_PROJECT_DIR = '/content/drive/MyDrive/SANTA_2025'
os.makedirs(DRIVE_PROJECT_DIR, exist_ok=True)
os.makedirs(f'{DRIVE_PROJECT_DIR}/checkpoints', exist_ok=True)
os.makedirs(f'{DRIVE_PROJECT_DIR}/results', exist_ok=True)

print(f"✅ Project directory: {DRIVE_PROJECT_DIR}")

# Step 3: Clone/update repo
REPO_URL = 'YOUR_GITHUB_REPO_URL_HERE'  # <-- Change this!
LOCAL_DIR = '/content/SANTA'

if os.path.exists(LOCAL_DIR):
    # Pull latest
    os.chdir(LOCAL_DIR)
    !git pull
else:
    # Clone fresh
    !git clone {REPO_URL} {LOCAL_DIR}
    os.chdir(LOCAL_DIR)

# Step 4: Install dependencies
!pip install -q numpy numba scipy shapely cma tqdm matplotlib

# Step 5: Create symlinks for persistent checkpoints
import os

# Remove local checkpoint/results dirs if they exist
if os.path.exists('checkpoints') and not os.path.islink('checkpoints'):
    shutil.rmtree('checkpoints')
if os.path.exists('results') and not os.path.islink('results'):
    shutil.rmtree('results')

# Create symlinks to Drive
if not os.path.exists('checkpoints'):
    os.symlink(f'{DRIVE_PROJECT_DIR}/checkpoints', 'checkpoints')
if not os.path.exists('results'):
    os.symlink(f'{DRIVE_PROJECT_DIR}/results', 'results')

print(f"✅ Checkpoints symlinked to: {DRIVE_PROJECT_DIR}/checkpoints")
print(f"✅ Results symlinked to: {DRIVE_PROJECT_DIR}/results")

# Step 6: Check for existing checkpoints
existing = os.listdir(f'{DRIVE_PROJECT_DIR}/checkpoints')
if existing:
    print(f"\n📁 Found {len(existing)} existing checkpoints:")
    for f in sorted(existing)[-5:]:
        print(f"   {f}")
    print("\n   Will auto-resume from latest checkpoint!")

print("\n" + "="*60)
print("Setup complete! Now run the optimization:")
print("="*60)
print("""
# Quick test:
!python main.py --quick

# Full run with 24 workers (auto-resumes if crashed):
!python main.py --full --workers 24

# Check progress:
!python main.py --analyze

# Generate submission when done:
!python main.py --submit --output /content/drive/MyDrive/SANTA_2025/submission.csv
""")
