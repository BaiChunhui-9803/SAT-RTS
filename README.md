# SAT-RTS: A Systematic Framework for Tactical Knowledge Extraction and Visualization-Based Attribution Analysis in RTS Games

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

Efficient tactical knowledge extraction and analysis in real-time strategy (RTS) games are often constrained by high-dimensional coupled state-action data and the inherent opacity of black-box decision-making processes. SAT-RTS (State-Action-Tactic Analysis Pipeline) bridges the gap between raw data streams and actionable insights through a systematic framework designed for **interpretable tactical knowledge extraction** and **visualization-based attribution analysis**.

By integrating interpretable visualization tools with scientific analytical methods, SAT-RTS enables a **hierarchical understanding** of the deep-seated drivers behind critical decisions. The framework provides:

- **State-Stream Abstraction**: Cluster-centric BK-tree algorithm with multi-aspect similarity metrics for efficient data decoupling
- **Semantic Tactical Labels**: Rule-based multi-label extraction transforming unstructured sequences into discrete, interpretable knowledge
- **Policy Evaluation**: Intuitive visual evidence for assessing autonomous learning systems in complex dynamic environments
- **Transparency Enhancement**: Robust tools for revealing decision-making logic in black-box RL policies

### Key Features

- **Multi-Aspect Similarity Metrics**: Quantifies similarities across states, state-transition sequences, and action patterns through Adapted EMD and DTW-based sequence alignment
- **Cluster-Centric BK-Tree**: Efficient stream clustering algorithm for state-stream abstraction and data decoupling with O(n³) Hungarian-based distance computation
- **Adapted EMD Distance Metric**: Hungarian algorithm-based optimal unit matching with virtual point padding for inconsistent unit counts
- **Fitness Landscape Visualization**: MDS-based dimensionality reduction and linear interpolation for solution distribution and problem structure characterization
- **Action Sequence Pattern Mining**: Exhaustive search with minimum support threshold for tactical pattern extraction from high-fitness solutions
- **Rule-Based Multi-Label Tactic Extraction**: Transforms unstructured sequences into **discrete, semantic tactical labels**
- **Attribution Analysis**: Cross-analysis of state-tactic relationships with treemap visualization for tactical payoff correlation and policy evaluation

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Quick Start

```bash
# Clone the repository
git clone https://github.com/your-username/SAT-RTS.git
cd SAT-RTS

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
numpy>=1.20.0
scipy>=1.7.0
pandas>=1.3.0
scikit-learn>=0.24.0
matplotlib>=3.4.0
seaborn>=0.11.0
plotly>=5.3.0
networkx>=2.6.0
mlxtend>=0.19.0
pyfpgrowth>=1.0.0
POT>=0.8.0  # Optional: for optimal transport distances
```

## Project Structure

```
SAT-RTS/
├── config.py              # Centralized configuration management
├── main.py                # Main analysis entry point
├── requirements.txt       # Python dependencies
├── data/                  # Dataset storage
│   ├── MarineMicro_MvsM_4/    # 4v4 Marine scenario (sce1/sce1m)
│   ├── MarineMicro_MvsM_4_dist/  # Distributed 4v4 (sce2/sce2m)
│   ├── MarineMicro_MvsM_8/    # 8v8 Marine scenario (sce2/sce2m)
│   └── multi_alg/             # Multi-algorithm comparison data
├── src/
│   ├── analysis/          # Analysis modules
│   │   ├── bktree_builder.py     # Cluster-centric BK-tree construction
│   │   ├── streaming_clustering.py  # Online stream clustering algorithms
│   │   ├── pattern.py            # Action sequence pattern mining
│   │   ├── multi_analysis.py     # Multi-algorithm comparison
│   │   └── sensitivity.py        # Parameter sensitivity analysis
│   └── distance/          # Distance metric implementations
│       ├── base.py        # Adapted EMD based on Hungarian algorithm
│       ├── custom.py      # Custom RTS-specific distances
│       ├── emd.py         # Earth Mover's Distance
│       ├── hausdorff.py   # Hausdorff distance
│       ├── chamfer.py     # Chamfer distance
│       └── wasserstein.py # Wasserstein distance
├── output/                # Generated outputs
│   ├── sankey/            # Sankey diagrams for tactic flow
│   ├── fitness_landscape/ # 3D fitness landscapes with MDS
│   ├── clustering/        # Clustering results visualization
│   └── analysis/          # Attribution analysis reports
└── map/                   # StarCraft II map files (.SC2Map)
```

## Usage

### Basic Analysis

```python
from config import get_data_paths, DEFAULT_MAP_ID, DEFAULT_DATA_ID

# Configure dataset
map_id = DEFAULT_MAP_ID      # "MarineMicro_MvsM_4"
data_id = DEFAULT_DATA_ID    # "6"

# Get data paths
paths = get_data_paths(map_id, data_id)

# Run main analysis
python main.py
```

### Switching Datasets

Modify the configuration in your script:

```python
# Available datasets:
# - MarineMicro_MvsM_4 (data_ids: ["6"])
# - MarineMicro_MvsM_4_dist (data_ids: ["1"])
# - MarineMicro_MvsM_8 (data_ids: ["1"])

map_id = "MarineMicro_MvsM_8"
data_id = "1"
paths = get_data_paths(map_id, data_id)
```

### Multi-Algorithm Comparison

```python
from src.analysis.multi_analysis import SCENARIOS, CURRENT_SCENARIO

# Available scenarios: sce1, sce1m, sce2, sce2m
CURRENT_SCENARIO = "sce2"
```

## Configuration

The `config.py` module provides centralized path management:

```python
from config import (
    get_data_paths,      # Get all data paths for a dataset
    get_output_dir,      # Get output directory path
    get_cache_path,      # Get cache file path
    DEFAULT_MAP_ID,      # Default map identifier
    DEFAULT_DATA_ID,     # Default data identifier
)

# Example usage
paths = get_data_paths("MarineMicro_MvsM_4", "6")
print(paths["game_result_path"])  # path to game results
print(paths["action_log_path"])   # path to action logs
```

## Methodology

SAT-RTS implements a three-level hierarchical analysis pipeline for interpretable tactical knowledge extraction and data decoupling:

### 1. Battlefield Situation Similarity Assessment

- **Multi-Aspect Similarity Metrics**: Quantifies state similarity via Adapted EMD with optimal unit matching
- **Cluster-Centric BK-Tree**: Hierarchical metric tree for efficient **state-stream abstraction** and **data decoupling**
  - Primary clustering by unit distribution (coordinate distance $d_c$)
  - Secondary clustering by HP differences (health difference $d_h$)
- **Virtual Point Padding**: Handles inconsistent unit counts in RTS combat scenarios

### 2. State-Transition Sequence Similarity Assessment

- **DTW-Based Distance**: Dynamic Time Warping extended to RTS game state-transition sequences for non-linear alignment
- **Fitness Landscape Visualization**: 
  - MDS (Multidimensional Scaling) for dimensionality reduction
  - Linear interpolation for landscape approximation
  - Captures multi-modality, ruggedness, and funnel-like topography
- **State Value Landscape**: Bipolar visualization capturing structural features of state space

### 3. Tactic Similarity Assessment

- **Action Sequence Pattern Mining**: Exhaustive search with minimum support threshold ($\sigma$)
- **Rule-Based Multi-Label Extraction**: Transforms unstructured sequences into **discrete, semantic tactical labels**
- **Attribution Analysis**: State-tactic payoff correlation via treemap visualization for **policy evaluation**
- **Sankey Diagrams**: Visualize causal and temporal relationships in tactic flows
- **Transparency Enhancement**: Intuitive visual evidence for understanding autonomous learning systems

## Distance Metrics

SAT-RTS implements multiple distance metrics for battlefield situation similarity assessment:

| Metric | Distribution Awareness | Mismatch Robustness | Computational Efficiency |
|--------|------------------------|---------------------|--------------------------|
| **Adapted EMD (Ours)** | ++ | ++ | + |
| 1-Wasserstein (EMD) | ++ | + | - |
| 2-Wasserstein | ++ | + | - |
| Hausdorff Distance | -- | -- | ++ |
| Chamfer Distance | + | - | ++ |

### Adapted EMD Distance Metric

The adapted EMD distance metric addresses the combat unit matching problem in RTS game states through three key innovations:

1. **Optimal Assignment**: Transforms unit matching into a Linear Assignment Problem (LAP) solved via Hungarian algorithm
2. **Virtual Point Padding**: Handles inconsistent unit counts by adding virtual points at distance $D_v \gg \max_{i,j} d(a_i, b_j)$
3. **Dual Metrics**: Computes both coordinate distribution distance ($d_c$) and HP difference ($d_h$)

For states $S_A$ and $S_B$ with unit sets, the distance metrics are:

$$d_c = \sum_{i=1}^{m} \sum_{j=1}^{m} D^1_{ij} X^1_{ij} + \sum_{i=1}^{n} \sum_{j=1}^{n} D^2_{ij} X^2_{ij}$$

$$d_h = \sum_{i=1}^{m} \sum_{j=1}^{m} |H^1_{Ai} - H^1_{Bj}| X^1_{ij} + \sum_{i=1}^{n} \sum_{j=1}^{n} |H^2_{Ai} - H^2_{Bj}| X^2_{ij}$$

Where $X^1$ and $X^2$ are binary assignment matrices ensuring each unit is matched exactly once.

## Experimental Scenarios

Experiments are conducted on the **StarCraft Multi-Agent Challenge (SMAC)** platform with 8,100 complete combat simulations across multiple scenarios. SAT-RTS effectively identifies key tactical patterns and provides **intuitive visual evidence for policy evaluation**, demonstrating its effectiveness in enhancing the **transparency of autonomous learning systems** in complex dynamic environments.

| Scenario | Combat Scale | Mirror Scenario | Sampled States |
|----------|-------------|-----------------|----------------|
| sce1 | M4 vs M4 | sce1m | ~80,175 |
| sce2 | M8 vs M8 | sce2m | ~127,299 |

### Fitness Landscape Characteristics

RTS micromanagement fitness landscapes exhibit three key characteristics:

- **Multi-modality**: Multiple global optima representing diverse tactical combinations (focus-fire, kiting, decoy maneuvers)
- **Ruggedness**: High density of local optima amplifying search complexity due to sensitivity to tactical perturbations
- **Funnel-like Topography**: Global basin of attraction with pronounced gradient separating high-fitness from poor solutions

## Output Categories

| Category | Description |
|----------|-------------|
| `sankey/` | Sankey diagrams for action sequence and tactic pattern flow visualization |
| `fitness_landscape/standard/` | 2D/3D fitness landscapes with clustering visualization |
| `fitness_landscape/interpretability/` | Attribution analysis visualizations |
| `clustering/standard/` | Standard stream clustering results |
| `clustering/difficult/` | Edge case clustering analysis |
| `analysis/pattern/` | Sequential pattern mining results |
| `cache/npy/` | Cached distance matrices and MDS coordinates |
