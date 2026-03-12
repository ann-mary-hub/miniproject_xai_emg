$ErrorActionPreference = 'Stop'

# Clear caches (optional) - comment out if you want to keep cache
# Remove-Item cache\feature_cache_*.npz -ErrorAction SilentlyContinue

# Hard-coded settings to match main_driver defaults
$env:PAPER_LOCK='0'
$env:PAPER_EXACT_MODE='1'
$env:PAPER_USE_ALL_FEATURES='0'
$env:TOP_K='15'

$env:USE_WINDOW_SEGMENTATION='1'
$env:WINDOW_SEC='1.0'
$env:WINDOW_OVERLAP='0.0'

$env:USE_DETERMINISTIC_SPLIT='0'
$env:USE_SMOTE='1'
$env:SMOTE_SCOPE='all'

$env:RUN_EXPLAINABILITY='1'
$env:TRAIN_MODEL='0'
$env:SKIP_CV='0'

python main_driver.py
