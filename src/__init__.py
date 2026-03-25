"""
SAT-RTS project configuration

Used for centralized project configuration management.
"""

__version__ = "1.0.0"
__author__ = "SAT-RTS Team"

# Project root directory
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
CACHE_DIR = OUTPUT_DIR / "cache"

# Output directory constants
OUTPUT_SANKEY = "sankey"
OUTPUT_SANKEY_D = "sankey/doubled"
OUTPUT_SANKEY_TACTIC = "sankey/tactic"
OUTPUT_SANKEY_TACTIC_D = "sankey/tactic_d"
OUTPUT_FITNESS_STANDARD = "fitness_landscape/standard"
OUTPUT_FITNESS_INTERP = "fitness_landscape/interpretability"
OUTPUT_CLUSTERING_STANDARD = "clustering/standard"
OUTPUT_CLUSTERING_INCONSISTENT = "clustering/inconsistent"
OUTPUT_CLUSTERING_DIFFICULT = "clustering/difficult"
OUTPUT_ANALYSIS_PATTERN = "analysis/pattern"
OUTPUT_CACHE_NPY = "cache/npy"
