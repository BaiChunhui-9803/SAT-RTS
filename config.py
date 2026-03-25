# config.py
"""
Project Configuration File - Centralized management of all path configurations

Usage:
    from config import get_data_paths, get_output_dir, DEFAULT_MAP_ID, DEFAULT_DATA_ID

    # Get data paths
    paths = get_data_paths("MarineMicro_MvsM_4", "6")
    distance_matrix_folder = paths["distance_matrix_folder"]

    # Get output path
    output_dir = get_output_dir("sankey", "tactic")
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
CACHE_DIR = OUTPUT_DIR / "cache"


def get_data_paths(map_id: str, data_id: str) -> dict:
    """
    Get all data paths for specified map_id and data_id

    Args:
        map_id: Map ID, e.g., "MarineMicro_MvsM_4"
        data_id: Data ID, e.g., "6"

    Returns:
        dict: Dictionary containing all paths
    """
    base_path = DATA_DIR / map_id / data_id

    return {
        "distance_matrix_folder": str(base_path / "distance"),
        "primary_bktree_path": str(base_path / "bktree" / "primary_bktree.json"),
        "secondary_bktree_prefix": str(base_path / "bktree" / "secondary_bktree"),
        "state_node_path": str(base_path / "graph" / "state_node.txt"),
        "node_log_path": str(base_path / "graph" / "node_log.txt"),
        "game_result_path": str(base_path / "game_result.txt"),
        "action_log_path": str(base_path / "action_log.csv"),
        "action_path": str(base_path / "sub_q_table"),
    }


class OutputPaths:
    """Output Path Manager"""

    SANKEY = "sankey"
    SANKEY_D = "sankey/doubled"
    SANKEY_TACTIC = "sankey/tactic"
    SANKEY_TACTIC_D = "sankey/tactic_d"
    FITNESS_STANDARD = "fitness_landscape/standard"
    FITNESS_INTERP = "fitness_landscape/interpretability"
    CLUSTERING_STANDARD = "clustering/standard"
    CLUSTERING_INCONSISTENT = "clustering/inconsistent"
    CLUSTERING_DIFFICULT = "clustering/difficult"
    ANALYSIS_PATTERN = "analysis/pattern"
    CACHE_NPY = "cache/npy"
    TACTICS_DICT = "tactics_dict"

    @classmethod
    def get(
        cls, category: str, map_id: str = "", data_id: str = "", create: bool = True
    ) -> str:
        """
        Get output directory path

        Args:
            category: Output category (e.g., cls.SANKEY, cls.FITNESS_STANDARD)
            map_id: Map ID (optional)
            data_id: Data ID (optional)
            create: Whether to automatically create directory

        Returns:
            str: Output directory path
        """
        path = OUTPUT_DIR / category
        if map_id:
            path = path / map_id
        if data_id:
            path = path / data_id

        if create:
            path.mkdir(parents=True, exist_ok=True)

        return str(path)

    @classmethod
    def get_file(
        cls, category: str, filename: str, map_id: str = "", data_id: str = ""
    ) -> str:
        """
        Get full path for output file

        Args:
            category: Output category
            filename: File name
            map_id: Map ID (optional)
            data_id: Data ID (optional)

        Returns:
            str: Full output file path
        """
        dir_path = cls.get(category, map_id, data_id)
        return str(Path(dir_path) / filename)


def get_output_dir(category: str, map_id: str = "", data_id: str = "") -> str:
    """
    Get output directory (convenience function)

    Args:
        category: Output category (e.g., "sankey", "fitness_landscape/standard")
        map_id: Map ID (optional)
        data_id: Data ID (optional)

    Returns:
        str: Output directory path
    """
    return OutputPaths.get(category, map_id, data_id)


def get_cache_path(filename: str) -> str:
    """
    Get cache file path

    Args:
        filename: Cache file name (e.g., "log_positions_MarineMicro_MvsM_4_6.npy")

    Returns:
        str: Full cache file path
    """
    cache_path = CACHE_DIR / "npy"
    cache_path.mkdir(parents=True, exist_ok=True)
    return str(cache_path / filename)


DEFAULT_MAP_ID = "MarineMicro_MvsM_4"
DEFAULT_DATA_ID = "6"

AVAILABLE_DATASETS = {
    "MarineMicro_MvsM_4": {
        "description": "4v4 Marine Battle",
        "data_ids": ["6"],
    },
    "MarineMicro_MvsM_4_dist": {
        "description": "4v4 Marine Battle (Distributed)",
        "data_ids": ["1"],
    },
    "MarineMicro_MvsM_8": {
        "description": "8v8 Marine Battle",
        "data_ids": ["1"],
    },
}

# Multi-algorithm comparison dataset configuration
MULTI_ALG_DATASETS = {
    "sce-1": {"description": "Scenario 1 - MarineMicro_MvsM_4"},
    "sce-1m": {"description": "Scenario 1 Mirror - MarineMicro_MvsM_4_mirror"},
    "sce-2": {"description": "Scenario 2 - MarineMicro_MvsM_4_dist"},
    "sce-2m": {"description": "Scenario 2 Mirror - MarineMicro_MvsM_4_dist_mirror"},
    "sce-3": {"description": "Scenario 3 - MarineMicro_MvsM_8"},
    "sce-3m": {"description": "Scenario 3 Mirror - MarineMicro_MvsM_8_mirror"},
}


def get_multi_alg_path(sce_id: str, filename: str = "data.json") -> str:
    """
    Get multi-algorithm comparison data path

    Args:
        sce_id: Scenario ID (e.g., "sce-1", "sce-2m")
        filename: File name (default: data.json)

    Returns:
        str: Full data file path
    """
    return str(DATA_DIR / "multi_alg" / sce_id / filename)


def get_multi_alg_output_path(sce_id: str, subdir: str = "") -> str:
    """
    Get multi-algorithm comparison output path

    Args:
        sce_id: Scenario ID
        subdir: Subdirectory name

    Returns:
        str: Output directory path
    """
    path = OUTPUT_DIR / "multi_alg" / sce_id
    if subdir:
        path = path / subdir
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def list_available_datasets() -> dict:
    """List all available datasets"""
    result = {}
    for map_id, info in AVAILABLE_DATASETS.items():
        map_path = DATA_DIR / map_id
        if map_path.exists():
            available_data_ids = []
            for data_id in info["data_ids"]:
                if (map_path / data_id).exists():
                    available_data_ids.append(data_id)
            if available_data_ids:
                result[map_id] = {
                    "description": info["description"],
                    "available_data_ids": available_data_ids,
                }
    return result


if __name__ == "__main__":
    print("=" * 50)
    print("SAT-RTS Project Configuration")
    print("=" * 50)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print()

    print("Default Data Paths:")
    paths = get_data_paths(DEFAULT_MAP_ID, DEFAULT_DATA_ID)
    for key, value in paths.items():
        exists = "Y" if Path(value).exists() else "N"
        print(f"  [{exists}] {key}")
    print()

    print("Available Datasets:")
    datasets = list_available_datasets()
    if datasets:
        for map_id, info in datasets.items():
            print(f"  - {map_id}: {info['description']}")
            print(f"    Available Data IDs: {', '.join(info['available_data_ids'])}")
    else:
        print("  No datasets available")
