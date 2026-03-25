"""
multi-algorithm comparative analysis module

comparison HRL_IMCBS and QMIX, COMA, IQL, QTRAN, VDN performance of algorithms
"""

import json
import math
import os
import time
from collections import defaultdict
import pandas as pd
import seaborn as sns
from sklearn.manifold import MDS, TSNE
from scipy.stats import norm
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.pyplot as plt

# use project internal modules
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.distance.base import CustomDistance
from src.analysis.bktree_builder import BKTree, ClusterNode

from scipy.interpolate import griddata
from scipy.stats import gaussian_kde, skewnorm
import random
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans

# import config
from config import get_data_paths, get_output_dir, OutputPaths, DATA_DIR

custom_distance = CustomDistance(threshold=0.5)

from joblib import Parallel, delayed
import multiprocessing
from multiprocessing import Pool, shared_memory
from functools import partial
from sklearn.metrics import silhouette_score
from typing import List, Tuple, Callable, Dict
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox
from matplotlib.ticker import FuncFormatter

# path manager - use project config
path_manager = {"distance_matrix_folder": get_output_dir(OutputPaths.FITNESS_STANDARD)}

# sampling_manager = {
#     "hrl_imcbs": 10,  # partial sampling
#     "QMIX": 100,
#     "COMA": 100,
#     "IQL": 100,
#     "QTRAN": 100,
#     "VDN": 100,
# }

# ============================================================
# global_sample_size = 9999999
global_sample_size = 100
global_band_height = 1

global_primary_threshold = 1.0
global_secondary_threshold = 0.5
# global_primary_threshold = 3.0
# global_secondary_threshold = 1.5

ENABLE = {
    "HRL_IMCBS": True,
    # "QMIX": True,
    # "COMA": True,
    # "IQL": True,
    # "QTRAN": True,
    # "VDN": True,
}
# ------------------------------------------------------------
# scene config dictionary - all available experiment scenes
# ------------------------------------------------------------
SCENARIOS = {
    "sce1": {
        "map": "MarineMicro_MvsM_4",
        "HRL_IMCBS": "6",
        "QMIX": "bktree_3",
        "COMA": "bktree_7",
        "IQL": "bktree_7",
        "QTRAN": "bktree_3",
        "VDN": "bktree_3",
    },
    "sce1m": {
        "map": "MarineMicro_MvsM_4_mirror",
        "HRL_IMCBS": "3",
        "QMIX": "bktree_12",
        "COMA": "bktree_14",
        "IQL": "bktree_13",
        "QTRAN": "bktree_14",
        "VDN": "bktree_14",
    },
    "sce2": {
        "map": "MarineMicro_MvsM_4_dist",
        "HRL_IMCBS": "1",
        "QMIX": "bktree_17",
        "COMA": "bktree_19",
        "IQL": "bktree_20",
        "QTRAN": "bktree_22",
        "VDN": "bktree_23",
    },
    "sce3": {
        "map": "MarineMicro_MvsM_8",
        "HRL_IMCBS": "1",
        "QMIX": "bktree_24",
        "COMA": "bktree_18",
        "IQL": "bktree_11",
        "QTRAN": "bktree_19",
        "VDN": "bktree_20",
    },
    "sce2m": {
        "map": "MarineMicro_MvsM_4_dist_mirror",
        "HRL_IMCBS": "3",
        "QMIX": "bktree_15",
        "COMA": "bktree_14",
        "IQL": "bktree_14",
        "QTRAN": "bktree_16",
        "VDN": "bktree_17",
    },
    "sce3m": {
        "map": "MarineMicro_MvsM_8_mirror",
        "HRL_IMCBS": "1",
        "QMIX": "bktree_21",
        "COMA": "bktree_25",
        "IQL": "bktree_24",
        "QTRAN": "bktree_23",
        "VDN": "bktree_22",
    },
}

# ------------------------------------------------------------
# currently active scene - modify here to switch scene
# optionalvalue: "sce1", "sce1m", "sce2", "sce3", "sce2m", "sce3m"
# ------------------------------------------------------------
CURRENT_SCENARIO = "sce1"

scenario_manager = SCENARIOS[CURRENT_SCENARIO]

# ============================================================
# data manager - use project configofrelative path
# note：only if enabled(ENABLE)algorithm path will be used

data_manager = {
    "HRL_IMCBS": get_data_paths(scenario_manager["map"], scenario_manager["HRL_IMCBS"]),
    # following algorithm data needs migration to data/ can be enabled after directory
    # "QMIX": _get_algo_data_paths(scenario_manager['map'], scenario_manager['QMIX']),
    # "COMA": _get_algo_data_paths(scenario_manager['map'], scenario_manager['COMA']),
    # "IQL": _get_algo_data_paths(scenario_manager['map'], scenario_manager['IQL']),
    # "QTRAN": _get_algo_data_paths(scenario_manager['map'], scenario_manager['QTRAN']),
    # "VDN": _get_algo_data_paths(scenario_manager['map'], scenario_manager['VDN']),
}

scenario_manager = {
    k: v for k, v in scenario_manager.items() if k == "map" or ENABLE.get(k, False)
}

# 2. filter data_manager
data_manager = {k: v for k, v in data_manager.items() if ENABLE.get(k, False)}

# ------- algorithmmatchcolor table -------
ALGO_COLOR = {
    "HRL_IMCBS": "red"  # forced red
    # 'HRL_IMCBS': '#2CA02C'          # forced red
}
# other algorithms cycle through colors
COLOR_LIST = [
    "#1F77B4",
    "#2CA02C",
    "#FF7F0E",
    "#9467BD",
    "#17BECF",
    # '#1F77B4', 'red', '#FF7F0E', '#9467BD', '#17BECF'
]

ALL_ALGORITHMS = [
    "HRL_IMCBS",
    "QMIX",
    "COMA",
    "IQL",
    "QTRAN",
    "VDN",
]  # by ENABLE orderalso works
GLOBAL_COLOR_MAP = {"HRL_IMCBS": ALGO_COLOR["HRL_IMCBS"]}  # first put forced red in

UNIT_MAX_HP = 45.0

# from COLOR_LIST ordertake remaining 5 items
for i, algo in enumerate(ALL_ALGORITHMS):
    if algo not in GLOBAL_COLOR_MAP:
        GLOBAL_COLOR_MAP[algo] = COLOR_LIST[i - 1]  # i-1 because HRL_IMCBS already occupied 0


class BKTreeNode:
    def __init__(self, state, cluster_id):
        self.state = state
        self.cluster_id = cluster_id
        self.children = {}

    def add_child(self, dist, node):
        self.children[dist] = node


class BKTree:
    def __init__(self):
        self.root = None

    def find_node_by_cluster_id(self, cluster_id):
        """
        Recursively find BKTreeNode with specified cluster_id
        """

        def search_node(node):
            if node.cluster_id == cluster_id:
                return node
            for child in node.children.values():
                result = search_node(child)
                if result:
                    return result
            return None

        if self.root:
            return search_node(self.root)
        return None


# def get_norm_state(state_str):


def load_bk_tree_from_file(file_path):
    """
    Load BKTree data from file and restore as BKTree instance

    :param file_path: File path
    :return: BKTree instance
    """

    def deserialize_node(node_data):
        """
        recursive desequenceize nodes
        """
        state = {"state": [node_data["state"]]}
        node = BKTreeNode(state, node_data["cluster_id"])
        for dist, child_data in node_data["children"].items():
            child_node = deserialize_node(child_data)
            node.add_child(float(dist), child_node)
        return node

    with open(file_path, "r") as file:
        tree_data = json.load(file)

    bk_tree = BKTree()
    bk_tree.root = deserialize_node(tree_data)
    return bk_tree


def find_max_cluster_id(node, max_cluster_id):
    """
    Recursively find maximum cluster_id in BKTree
    """
    if node.cluster_id > max_cluster_id[0]:
        max_cluster_id[0] = node.cluster_id

    for child in node.children.values():
        find_max_cluster_id(child, max_cluster_id)


def get_max_cluster_id(bk_tree):
    """
    Get maximum cluster_id in BKTree
    """
    max_cluster_id = [0]  # use list to store max values，for recursive modification
    if bk_tree.root:
        find_max_cluster_id(bk_tree.root, max_cluster_id)
    return max_cluster_id[0]


def classify_new_state(new_state, bktree, threshold=1.0):
    cluster_id = bktree.query(new_state, threshold)
    if cluster_id is not None:
        return cluster_id
    else:
        new_cluster_id = bktree.get_next_cluster_id()
        new_node = ClusterNode(new_state, new_cluster_id)
        bktree.insert(new_node, bktree.root)
        return new_cluster_id


def get_state_cluster(primary_bktree, secondary_bktree, norm_state):
    def _get_state_value(norm_state):
        state_list = norm_state.get("state", [])
        if not state_list:
            return 0.0

        diffs = []
        for item in state_list:
            blue_sum = sum(vec[-1] for vec in item.get("blue_army", []) if vec)
            red_sum = sum(vec[-1] for vec in item.get("red_army", []) if vec)
            diffs.append(red_sum - blue_sum)

        return round(sum(diffs) / len(diffs) * UNIT_MAX_HP, 2) if diffs else 0.0

    state_value = _get_state_value(norm_state)

    if primary_bktree.root is None:
        primary_bktree.root = ClusterNode(norm_state, 1)
        secondary_bktree[1].root = ClusterNode(norm_state, 1)
        # self.primary_bktree.root.state_list = [norm_state]
        return (1, 1), 0.0
    else:
        new_cluster_id = classify_new_state(
            norm_state, primary_bktree, threshold=global_primary_threshold
        )
        # new_cluster_id = classify_new_state(norm_state, primary_bktree, threshold=3.0)
        if secondary_bktree[new_cluster_id].root is None:
            secondary_bktree[new_cluster_id].root = ClusterNode(norm_state, 1)
            return (new_cluster_id, 1), state_value
        else:
            new_sub_cluster_id = classify_new_state(
                norm_state,
                secondary_bktree[new_cluster_id],
                threshold=global_secondary_threshold,
            )
            # new_sub_cluster_id = classify_new_state(norm_state, secondary_bktree[new_cluster_id], threshold=1.5)
            return (new_cluster_id, new_sub_cluster_id), state_value


def save_distance_matrix(matrix, file_path):
    """
    saveDistance matrixto file
    :param matrix: Distance matrix
    :param file_path: File path
    """
    np.save(file_path, matrix)


def load_distance_matrix(file_path):
    """
    load from fileDistance matrix
    :param file_path: File path
    :return: Distance matrix
    """
    return np.load(file_path)


def calculate_distance_matrix(reverse_dict, custom_distance, secondary_bk_trees):
    """
    Calculate distance matrix
    :param reverse_dict: Reverse dictionary
    :param custom_distance: Custom distance calculation function
    :param secondary_bk_trees: Secondary BKTree dictionary
    :return: Distance matrix
    """
    # Get the number of all clusters
    num_clusters = len(reverse_dict)

    # initializeDistance matrix
    # distance_matrix = np.zeros((num_clusters, num_clusters), dtype=np.float32)
    distance_matrix = np.zeros((num_clusters, num_clusters), dtype=np.float32)

    # # === unique modification start ===
    # import os
    # mmap_path = 'distance_matrix.memmap'
    # if os.path.exists(mmap_path):
    #     os.remove(mmap_path)
    # distance_matrix = np.memmap(mmap_path, dtype=np.float32, mode='w+',
    #                             shape=(num_clusters, num_clusters))
    # # === unique modification end ===

    # Get states of all clusters
    clusters = list(reverse_dict.values())

    # Initialize last output time
    last_output_time = time.time()

    # Initialize progress threshold
    progress_threshold = 0.01  # 10%

    # Calculate distance between each pair of clusters
    for i in range(num_clusters):
        for j in range(i + 1, num_clusters):  # Start from i+1 to avoid duplicate calculations
            state1 = clusters[i]["cluster"]
            state2 = clusters[j]["cluster"]

            # Get nodes of two states
            node1 = (
                secondary_bk_trees[state1[0]].find_node_by_cluster_id(state1[1]).state
            )
            node2 = (
                secondary_bk_trees[state2[0]].find_node_by_cluster_id(state2[1]).state
            )

            # Calculate distance
            dist = custom_distance.multi_distance(node1, node2)

            # Calculate Euclidean distance
            euclidean_distance = math.sqrt(dist[0] ** 2 + dist[1] ** 2)

            # fillDistance matrix
            distance_matrix[i, j] = euclidean_distance
            distance_matrix[j, i] = euclidean_distance

        # Check if progress output is needed
        current_time = time.time()
        progress = (i + 1) / num_clusters
        if progress >= progress_threshold or i == num_clusters - 1:
            time_elapsed = current_time - last_output_time
            print(
                f"Processed {i + 1} out of {num_clusters} states ({progress * 100:.1f}%) (Time elapsed: {time_elapsed:.2f} seconds)"
            )
            last_output_time = current_time
            progress_threshold += 0.01  # update progress threshold

    # distance on diagonal is 0
    for i in range(num_clusters):
        distance_matrix[i, i] = 0

    return distance_matrix


def dtw_distance(seq1, seq2, distance_matrix):
    """
    Calculate DTW distance between two sequence
    :param seq1: First sequence
    :param seq2: Second sequence
    :param distance_matrix: Distance matrix
    :return: DTWdistance
    """
    m = len(seq1)
    n = len(seq2)
    dtw_matrix = np.zeros((m + 1, n + 1))
    dtw_matrix[0, 0] = 0
    for i in range(1, m + 1):
        dtw_matrix[i, 0] = np.inf
    for j in range(1, n + 1):
        dtw_matrix[0, j] = np.inf

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = distance_matrix[seq1[i - 1], seq2[j - 1]]
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j - 1]
            )

    return dtw_matrix[m, n]


def calculate_dtw_distance_matrix(state_log, distance_matrix):
    """
    calculateallsequencebetweenofDTWDistance matrix
    :param state_log: contain allsequenceoflist
    :param distance_matrix: Distance matrix
    :return: DTWDistance matrix
    """
    num_sequences = len(state_log)
    dtw_distance_matrix = np.zeros((num_sequences, num_sequences))

    # Initialize last output time
    last_output_time = time.time()

    # Initialize progress threshold
    progress_threshold = 0.01  # 1%

    # calculateeachforsequencebetweenofDTWdistance
    for i in range(num_sequences):
        for j in range(i + 1, num_sequences):
            seq1 = state_log[i]
            seq2 = state_log[j]
            dtw_dist = dtw_distance(seq1, seq2, distance_matrix)
            dtw_distance_matrix[i, j] = dtw_dist
            dtw_distance_matrix[j, i] = dtw_dist

        # Check if progress output is needed
        current_time = time.time()
        progress = (i + 1) / num_sequences
        if progress >= progress_threshold or i == num_sequences - 1:
            time_elapsed = current_time - last_output_time
            print(
                f"Processed {i + 1} out of {num_sequences} logs ({progress * 100:.1f}%) (Time elapsed: {time_elapsed:.2f} seconds)"
            )
            last_output_time = current_time
            progress_threshold += 0.01  # update progress threshold

    return dtw_distance_matrix


def calculate_and_save_distance_matrix(
    reverse_dict, custom_distance, secondary_bk_trees, distance_matrix_folder
):
    """
    Calculate distance matrixand saveto file
    :param reverse_dict: Reverse dictionary
    :param custom_distance: Custom distance calculation function
    :param secondary_bk_trees: Secondary BKTree dictionary
    :param distance_matrix_folder: Distance matrixsave folder path
    :return: Distance matrix
    """
    # Ensure folder exists
    if not os.path.exists(distance_matrix_folder):
        os.makedirs(distance_matrix_folder)
        print(f"Created directory: {distance_matrix_folder}")

    # define matrixFile path
    suffix = "_".join(
        [f"{key.lower()}_{value}" for key, value in scenario_manager.items()]
    )
    state_distance_matrix_path = os.path.join(
        distance_matrix_folder, f"state_distance_matrix_{suffix}.npy"
    )

    # Check if matrix file exists
    if os.path.exists(state_distance_matrix_path):
        print(f"Loading state distance matrix from {state_distance_matrix_path}")
        return load_distance_matrix(state_distance_matrix_path)
    else:
        print("Calculating state distance matrix...")
        distance_matrix = calculate_distance_matrix(
            reverse_dict, custom_distance, secondary_bk_trees
        )
        print(f"Saving state distance matrix to {state_distance_matrix_path}")
        save_distance_matrix(distance_matrix, state_distance_matrix_path)
        return distance_matrix


def calculate_and_save_dtw_distance_matrix(
    state_log, distance_matrix, dtw_distance_matrix_folder, threshold=None
):
    """
    calculateDTWDistance matrixand saveto file
    :param state_log: contain allsequenceoflist
    :param distance_matrix: Distance matrix
    :param dtw_distance_matrix_folder: DTWDistance matrixsave folder path
    :return: DTWDistance matrix
    """
    # Ensure folder exists
    if not os.path.exists(dtw_distance_matrix_folder):
        os.makedirs(dtw_distance_matrix_folder)
        print(f"Created directory: {dtw_distance_matrix_folder}")

    # define matrixFile path
    suffix_1 = "".join(
        [f"{key.lower()}_{value}" for key, value in scenario_manager.items()]
    )
    suffix_2 = f"sample{global_sample_size}"
    # suffix_2 = "".join([f"{key}_{value}" for key, value in sampling_manager.items()])
    suffix = f"{suffix_1}_{suffix_2}"
    # define matrixFile path
    if threshold is not None:
        suffix += f"_threadhold_{threshold}"
    else:
        suffix += "_threadhold_none"
    log_distance_matrix_path = os.path.join(
        dtw_distance_matrix_folder, f"log_distance_matrix_{suffix}.npy"
    )

    # Check if matrix file exists
    if os.path.exists(log_distance_matrix_path):
        print(f"Loading DTW distance matrix from {log_distance_matrix_path}")
        return load_distance_matrix(log_distance_matrix_path)
    else:
        print("Calculating DTW distance matrix...")
        dtw_distance_matrix = calculate_dtw_distance_matrix(state_log, distance_matrix)
        print(f"Saving DTW distance matrix to {log_distance_matrix_path}")
        save_distance_matrix(dtw_distance_matrix, log_distance_matrix_path)
        return dtw_distance_matrix


# ---------- 1. collect information ----------
def build_state_algorithm_info(
    all_logs, all_results, all_data_set_ids, reverse_dict, dist_matrix
):
    """
    return state_info: dict[panorama_state_id -> dict]
    """
    n_states = len(reverse_dict)
    state_info = {
        sid: {
            "algorithms": set(),
            "freq_total": 0,
            "is_algo_opt": defaultdict(bool),
            "is_global_opt": False,
        }
        for sid in range(n_states)
    }

    # ---- 1. first find each algorithm optimal value & global optimumvalue ----
    algo_best = defaultdict(lambda: -np.inf)  # algorithm -> optimalresult
    algo_opt_logs = defaultdict(list)  # algorithm -> optimal log_id list（allow multiple tied optima）
    global_best = -np.inf
    global_opt_logs = []  # global optimum log_id list

    for log_id, (log, res, algo) in enumerate(
        zip(all_logs, all_results, all_data_set_ids)
    ):
        if res > algo_best[algo]:
            algo_best[algo] = res
            algo_opt_logs[algo] = [log_id]
        elif res == algo_best[algo]:
            algo_opt_logs[algo].append(log_id)

        if res > global_best:
            global_best = res
            global_opt_logs = [log_id]
        elif res == global_best:
            global_opt_logs.append(log_id)

    # ---- 2. fill information ----
    for log_id, (log, res, algo) in enumerate(
        zip(all_logs, all_results, all_data_set_ids)
    ):
        for state_id in log:
            info = state_info[state_id]
            info["algorithms"].add(algo)
            info["freq_total"] += 1

        # optimal log marker
        if log_id in algo_opt_logs[algo]:
            for state_id in log:
                state_info[state_id]["is_algo_opt"][algo] = True

        if log_id in global_opt_logs:
            for state_id in log:
                state_info[state_id]["is_global_opt"] = True

    # supplement：recordglobal optimumstatecorrespondsofalgorithm
    for sid in range(n_states):
        if state_info[sid]["is_global_opt"]:
            state_info[sid]["global_algos"] = {
                algo for algo, opt in state_info[sid]["is_algo_opt"].items() if opt
            }
    return state_info


def _get_mds_coords_path(distance_matrix_folder: str) -> str:
    """generate full coordinate file path"""
    map_name = scenario_manager["map"]
    enabled_algos = [a for a in ALL_ALGORITHMS if ENABLE.get(a, False)]
    algo_suffix = "_".join(a.lower() for a in enabled_algos)
    file_name = f"{map_name}_{algo_suffix}_mds_coords.npy"
    return os.path.join(distance_matrix_folder, file_name)


def get_or_create_mds_coords(
    distance_matrix: np.ndarray, distance_matrix_folder: str, random_state: int = 42
) -> np.ndarray:
    """
    if coord file exists then read directly；otherwise use MDS calculateand save。
    """
    coords_path = _get_mds_coords_path(distance_matrix_folder)
    if os.path.exists(coords_path):
        print(f"[MDS] load coords from {coords_path}")
        return np.load(coords_path)
    print("[MDS] fit & save coords ...")
    reducer = MDS(
        n_components=2, dissimilarity="precomputed", random_state=random_state, n_init=4
    )
    coords = reducer.fit_transform(distance_matrix)
    os.makedirs(distance_matrix_folder, exist_ok=True)
    np.save(coords_path, coords)
    print(f"[MDS] coords saved to {coords_path}")
    return coords


# ---------- figure1：uniquestate ----------
def plot_unique_states(state_info, dist_matrix, reverse_dict, save_path=None):
    coords = get_or_create_mds_coords(
        dist_matrix, path_manager["distance_matrix_folder"]
    )
    plt.figure(figsize=(7, 5))

    n_states = coords.shape[0]

    # 1. gray background：shared by multiple algorithms
    gray_mask = [
        sid for sid in range(n_states) if len(state_info[sid]["algorithms"]) != 1
    ]
    if gray_mask:
        gray_coords = coords[gray_mask]
        freq = np.array([state_info[sid]["freq_total"] for sid in gray_mask])
        alpha_gray = 0.2 + 0.5 * (freq - freq.min()) / (freq.max() - freq.min() + 1e-9)
        plt.scatter(
            gray_coords[:, 0],
            gray_coords[:, 1],
            c="#999999",
            s=15,
            alpha=0.3,
            label="common states",
        )

    # 2. monochrome algorithm（unique algorithm）
    for algo in ALL_ALGORITHMS:
        if algo not in ENABLE or not ENABLE[algo]:
            continue
        mask = [
            sid for sid in range(n_states) if state_info[sid]["algorithms"] == {algo}
        ]
        if not mask:
            continue
        color = GLOBAL_COLOR_MAP[algo]  # strict lookup
        plt.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=color,
            s=35,
            alpha=0.5,
            label=f"{algo} unique",
        )

    # 4. enlarge canvas（only current figure）
    fig = plt.gcf()
    fig.set_size_inches(10, 7)  # width 10.5 height 7.5
    # 5. legend：right side external + semi-transparent white background
    plt.legend(loc="upper left", fontsize=10, framealpha=1, fancybox=True, shadow=True)
    plt.title("Algorithm-specific unique states")
    plt.xlabel("MDS-1")
    plt.ylabel("MDS-2")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path + ".pdf", dpi=600)
    # plt.show()


# ---------- figure2：uniquestate + optimalheightbright ----------
def plot_unique_states_with_highlight(
    state_info, dist_matrix, reverse_dict, save_path=None
):
    coords = get_or_create_mds_coords(
        dist_matrix, path_manager["distance_matrix_folder"]
    )
    plt.figure(figsize=(7, 5))

    n_states = coords.shape[0]

    # 1. gray background：shared by multiple algorithms
    gray_mask = [
        sid for sid in range(n_states) if len(state_info[sid]["algorithms"]) != 1
    ]
    if gray_mask:
        gray_coords = coords[gray_mask]
        freq = np.array([state_info[sid]["freq_total"] for sid in gray_mask])
        alpha_gray = 0.2 + 0.5 * (freq - freq.min()) / (freq.max() - freq.min() + 1e-9)
        plt.scatter(
            gray_coords[:, 0],
            gray_coords[:, 1],
            c="#999999",
            s=15,
            alpha=0.3,
            label="common states",
        )

    # 2. monochrome algorithm（unique algorithm）
    for algo in ALL_ALGORITHMS:
        if algo not in ENABLE or not ENABLE[algo]:
            continue
        mask = [
            sid for sid in range(n_states) if state_info[sid]["algorithms"] == {algo}
        ]
        if not mask:
            continue
        color = GLOBAL_COLOR_MAP[algo]  # strict lookup
        plt.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=color,
            s=35,
            alpha=0.5,
            label=f"{algo} unique",
        )

    # 3. global optimumstar（by ALL_ALGORITHMS order，ensure legend order）
    legend_done = set()
    for algo in ALL_ALGORITHMS:  # fixed order
        if algo not in ENABLE or not ENABLE[algo]:
            continue
        # only keep“thestateisglobal optimum **and** current algorithm brings it to optimum”ofstate
        mask = [
            sid
            for sid in range(n_states)
            if state_info[sid]["is_global_opt"] and state_info[sid]["is_algo_opt"][algo]
        ]
        if not mask:
            continue

        color = GLOBAL_COLOR_MAP[algo]
        label = algo + " best" if algo not in legend_done else None
        legend_done.add(algo)

        # legendemptystar
        plt.scatter([], [], c=color, s=150, marker="*", label=label)
        # actualscatter
        for sid in mask:
            plt.scatter(
                coords[sid, 0], coords[sid, 1], c=color, s=150, marker="*", zorder=5
            )

    # 4. enlarge canvas（only current figure）
    fig = plt.gcf()
    fig.set_size_inches(10, 7)  # width 10.5 height 7.5
    # 5. legend：right side external + semi-transparent white background
    plt.legend(loc="upper left", fontsize=10, framealpha=1, fancybox=True, shadow=True)
    plt.title("Algorithm-specific unique states and states in bests")
    plt.xlabel("MDS-1")
    plt.ylabel("MDS-2")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path + ".pdf", dpi=600)
    # plt.show()


# ---------- figure3：each algorithmoptimalparsestate ----------
def plot_algorithm_best_states(state_info, dist_matrix, reverse_dict, save_path=None):
    coords = get_or_create_mds_coords(
        dist_matrix, path_manager["distance_matrix_folder"]
    )
    plt.figure(figsize=(7, 5))

    n_states = coords.shape[0]

    # 1. gray background：nonoptimalstate
    other_mask = [
        sid
        for sid in range(n_states)
        if not any(state_info[sid]["is_algo_opt"].values())
    ]
    if other_mask:
        plt.scatter(
            coords[other_mask, 0], coords[other_mask, 1], c="#999999", s=15, alpha=0.3
        )

    # 2. each algorithmownofoptimalstate（circle）
    for algo in ALL_ALGORITHMS:
        if algo not in ENABLE or not ENABLE[algo]:
            continue
        mask = [
            sid
            for sid in range(n_states)
            if state_info[sid]["is_algo_opt"].get(algo, False)
            and not state_info[sid]["is_global_opt"]
        ]
        if not mask:
            continue
        color = GLOBAL_COLOR_MAP[algo]  # strict lookup
        freq = np.array([state_info[sid]["freq_total"] for sid in mask])
        alpha = 0.4 + 0.6 * (freq - freq.min()) / (freq.max() - freq.min() + 1e-9)
        plt.scatter(
            coords[mask, 0], coords[mask, 1], c=color, s=60, alpha=0.5, marker="o"
        )

    # 3. global optimumstar（by ALL_ALGORITHMS order，ensure legend order）
    legend_done = set()
    for algo in ALL_ALGORITHMS:  # fixed order
        if algo not in ENABLE or not ENABLE[algo]:
            continue
        # only keep“thestateisglobal optimum **and** current algorithm brings it to optimum”ofstate
        mask = [
            sid
            for sid in range(n_states)
            if state_info[sid]["is_global_opt"] and state_info[sid]["is_algo_opt"][algo]
        ]
        if not mask:
            continue

        color = GLOBAL_COLOR_MAP[algo]
        label = algo + " best" if algo not in legend_done else None
        legend_done.add(algo)

        # legendemptystar
        plt.scatter([], [], c=color, s=150, marker="*", label=label)
        # actualscatter
        for sid in mask:
            plt.scatter(
                coords[sid, 0], coords[sid, 1], c=color, s=150, marker="*", zorder=5
            )

    # 4. enlarge canvas（only current figure）
    fig = plt.gcf()
    fig.set_size_inches(10, 7)  # width 10.5 height 7.5
    # 5. legend：right side external + semi-transparent white background
    plt.legend(loc="upper left", fontsize=10, framealpha=1, fancybox=True, shadow=True)
    plt.title("Algorithm-specific states in bests")
    plt.xlabel("MDS-1")
    plt.ylabel("MDS-2")
    # 6. adjustlayout，givenlegendreserve space
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path + ".pdf", dpi=600)
    # plt.show()


# ---------- figure4：FL（2-D a figure） ----------
def plot_FL(
    state_info, dist_matrix, reverse_dict, reverse_value_dict, save_path=None, show=True
):
    # 1. coordinates & mean
    coords = get_or_create_mds_coords(
        dist_matrix, path_manager["distance_matrix_folder"]
    )
    n_points = coords.shape[0]
    mean_vals = np.array([np.mean(reverse_value_dict[i]) for i in range(n_points)])

    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()

    # 2. interpolation grid
    xi = np.linspace(coords[:, 0].min(), coords[:, 0].max(), 100)
    yi = np.linspace(coords[:, 1].min(), coords[:, 1].max(), 100)
    grid_x, grid_y = np.meshgrid(xi, yi)
    grid_z = griddata(
        points=coords, values=mean_vals, xi=(grid_x, grid_y), method="nearest"
    )
    grid_z = np.nan_to_num(grid_z, nan=np.nanmean(mean_vals))

    # 3. colorbar 0 value
    min_v, max_v = mean_vals.min(), mean_vals.max()
    rng = max_v - min_v
    zero_pos = 0.5 if (min_v >= 0 or max_v <= 0) else (0 - min_v) / rng
    custom_colorscale = [[0, "#5c7ee6"], [zero_pos, "#ebebeb"], [1, "#b62d0a"]]

    fig = go.Figure()

    # 4. contour background color（fitness landscape）
    fig.add_trace(
        go.Contour(
            x=xi,
            y=yi,
            z=grid_z,
            colorscale=custom_colorscale,
            colorbar=dict(
                x=0.01,
                y=0.004,
                xanchor="right",
                yanchor="bottom",
                xref="paper",
                yref="paper",
                thickness=16,
                thicknessmode="pixels",
                len=0.992,
            ),
            contours=dict(coloring="heatmap"),
            showlegend=False,
            name="",
        )
    )

    # 3. plottingareaexpanded 20%（nonewhite border）
    pad_x = (x_max - x_min) * 0.02
    pad_y = (y_max - y_min) * 0.02

    # 8. layout
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Times New Roman", size=20, color="black"),
        # key：completely hide XY axis（with title、scaledegree、grid）
        xaxis=dict(
            range=[x_min - pad_x, x_max + pad_x], constrain="domain", visible=False
        ),
        yaxis=dict(
            range=[y_min - pad_y, y_max + pad_y], constrain="domain", visible=False
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,  # ← global legend off
    )

    # -------------- save & display --------------
    # fig.show()
    pdf_path = save_path if save_path.lower().endswith(".pdf") else save_path + ".pdf"
    fig.write_image(pdf_path, format="pdf", width=770, height=700, scale=1)


# ---------- figure5：uniquestate + FL（2-D a figure） ----------
def plot_unique_states_with_FL(
    state_info, dist_matrix, reverse_dict, reverse_value_dict, save_path=None, show=True
):
    # 1. coordinates & mean
    coords = get_or_create_mds_coords(
        dist_matrix, path_manager["distance_matrix_folder"]
    )
    n_points = coords.shape[0]
    mean_vals = np.array([np.mean(reverse_value_dict[i]) for i in range(n_points)])

    # 2. interpolation grid
    xi = np.linspace(coords[:, 0].min(), coords[:, 0].max(), 100)
    yi = np.linspace(coords[:, 1].min(), coords[:, 1].max(), 100)
    grid_x, grid_y = np.meshgrid(xi, yi)
    grid_z = griddata(
        points=coords, values=mean_vals, xi=(grid_x, grid_y), method="nearest"
    )
    grid_z = np.nan_to_num(grid_z, nan=np.nanmean(mean_vals))

    # 3. colorbar 0 value
    min_v, max_v = mean_vals.min(), mean_vals.max()
    rng = max_v - min_v
    zero_pos = 0.5 if (min_v >= 0 or max_v <= 0) else (0 - min_v) / rng
    custom_colorscale = [[0, "#5c7ee6"], [zero_pos, "#ebebeb"], [1, "#b62d0a"]]

    fig = go.Figure()

    # 4. contour background color（fitness landscape）
    fig.add_trace(
        go.Contour(
            x=xi,
            y=yi,
            z=grid_z,
            colorscale=custom_colorscale,
            colorbar=dict(
                x=0.046,
                y=0.05,
                xanchor="right",
                yanchor="bottom",
                xref="paper",
                yref="paper",
                thickness=16,
                thicknessmode="pixels",
                len=0.9,  # shorten by half，avoid touchinglegend
            ),
            contours=dict(coloring="heatmap"),
            showlegend=False,
            name="Fitness",
        )
    )

    # 5. scatter：shareoriginal logic color & label
    # 5.1 shared by multiple algorithms（gray background）
    gray_mask = [
        sid for sid in range(n_points) if len(state_info[sid]["algorithms"]) != 1
    ]
    if gray_mask:
        gray_coords = coords[gray_mask]
        freq = np.array([state_info[sid]["freq_total"] for sid in gray_mask])
        alpha_gray = 0.2 + 0.5 * (freq - freq.min()) / (freq.max() - freq.min() + 1e-9)
        fig.add_trace(
            go.Scatter(
                x=gray_coords[:, 0],
                y=gray_coords[:, 1],
                mode="markers",
                marker=dict(color="#999999", size=8, opacity=0.3),
                name="common states",
                showlegend=True,
            )
        )

    # 5.2 each algorithmonlyonestate
    for algo in ALL_ALGORITHMS:
        if algo not in ENABLE or not ENABLE[algo]:
            continue
        mask = [
            sid for sid in range(n_points) if state_info[sid]["algorithms"] == {algo}
        ]
        if not mask:
            continue
        color = GLOBAL_COLOR_MAP[algo]
        fig.add_trace(
            go.Scatter(
                x=coords[mask, 0],
                y=coords[mask, 1],
                mode="markers",
                marker=dict(color=color, size=10, opacity=0.8),
                name=f"{algo} unique",
                showlegend=True,
            )
        )

    # 8. layout
    fig.update_layout(
        # title='Algorithm-specific unique states & best + Fitness Landscape',
        xaxis_title="MDS-1",
        yaxis_title="MDS-2",
        width=1200,
        height=800,
        template="plotly_white",
        legend=dict(
            x=0.98,
            y=0.939,
            xanchor="right",
            yanchor="top",
            xref="paper",
            yref="paper",  # key：relative to paper edge
            bordercolor="Black",
            borderwidth=1,
            bgcolor="rgba(255,255,255,0.8)",
            traceorder="normal",
        ),
        font=dict(
            family="Times New Roman",
            size=20,  # optional
            color="black",
        ),
        # 5. plottingarea（axis）size no longer withlegendchange
        xaxis=dict(
            domain=[0.0, 0.78],
            visible=False,
            showticklabels=False,
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(
            domain=[0, 1],
            visible=False,
            showticklabels=False,
            showgrid=False,
            zeroline=False,
        ),
        margin=dict(l=0, r=0, t=0, b=0),  # l/b only leave 10 px anti-aliasing
    )

    # -------------- save & display --------------
    # fig.show()
    pdf_path = save_path if save_path.lower().endswith(".pdf") else save_path + ".pdf"
    fig.write_image(pdf_path, format="pdf", width=1200, height=800, scale=1)


# ---------- figure6：uniquestate + optimalheightbright + FL（2-D a figure） ----------
def plot_unique_states_with_highlight_with_FL(
    state_info, dist_matrix, reverse_dict, reverse_value_dict, save_path=None, show=True
):
    # 1. coordinates & mean
    coords = get_or_create_mds_coords(
        dist_matrix, path_manager["distance_matrix_folder"]
    )
    n_points = coords.shape[0]
    mean_vals = np.array([np.mean(reverse_value_dict[i]) for i in range(n_points)])

    # 2. interpolation grid
    xi = np.linspace(coords[:, 0].min(), coords[:, 0].max(), 100)
    yi = np.linspace(coords[:, 1].min(), coords[:, 1].max(), 100)
    grid_x, grid_y = np.meshgrid(xi, yi)
    grid_z = griddata(
        points=coords, values=mean_vals, xi=(grid_x, grid_y), method="nearest"
    )
    grid_z = np.nan_to_num(grid_z, nan=np.nanmean(mean_vals))

    # 3. colorbar 0 value
    min_v, max_v = mean_vals.min(), mean_vals.max()
    rng = max_v - min_v
    zero_pos = 0.5 if (min_v >= 0 or max_v <= 0) else (0 - min_v) / rng
    custom_colorscale = [[0, "#5c7ee6"], [zero_pos, "#ebebeb"], [1, "#b62d0a"]]

    fig = go.Figure()

    # 4. contour background color（fitness landscape）
    fig.add_trace(
        go.Contour(
            x=xi,
            y=yi,
            z=grid_z,
            colorscale=custom_colorscale,
            colorbar=dict(
                x=0.046,
                y=0.05,
                xanchor="right",
                yanchor="bottom",
                xref="paper",
                yref="paper",
                thickness=16,
                thicknessmode="pixels",
                len=0.9,  # shorten by half，avoid touchinglegend
            ),
            contours=dict(coloring="heatmap"),
            showlegend=False,
            name="Fitness",
        )
    )

    # 5. scatter：shareoriginal logic color & label
    # 5.1 shared by multiple algorithms（gray background）
    gray_mask = [
        sid for sid in range(n_points) if len(state_info[sid]["algorithms"]) != 1
    ]
    if gray_mask:
        gray_coords = coords[gray_mask]
        freq = np.array([state_info[sid]["freq_total"] for sid in gray_mask])
        alpha_gray = 0.2 + 0.5 * (freq - freq.min()) / (freq.max() - freq.min() + 1e-9)
        fig.add_trace(
            go.Scatter(
                x=gray_coords[:, 0],
                y=gray_coords[:, 1],
                mode="markers",
                marker=dict(color="#999999", size=8, opacity=0.3),
                name="common states",
                showlegend=True,
            )
        )

    # 5.2 each algorithmonlyonestate
    for algo in ALL_ALGORITHMS:
        if algo not in ENABLE or not ENABLE[algo]:
            continue
        mask = [
            sid for sid in range(n_points) if state_info[sid]["algorithms"] == {algo}
        ]
        if not mask:
            continue
        color = GLOBAL_COLOR_MAP[algo]
        fig.add_trace(
            go.Scatter(
                x=coords[mask, 0],
                y=coords[mask, 1],
                mode="markers",
                marker=dict(color=color, size=10, opacity=0.8),
                name=f"{algo} unique",
                showlegend=True,
            )
        )

    # 7. star：global optimum（by algorithm color）
    legend_done = set()
    for algo in ALL_ALGORITHMS:
        if algo not in ENABLE or not ENABLE[algo]:
            continue
        mask = [
            sid
            for sid in range(n_points)
            if state_info[sid]["is_global_opt"] and state_info[sid]["is_algo_opt"][algo]
        ]
        if not mask:
            continue
        color = GLOBAL_COLOR_MAP[algo]
        label = algo + " best" if algo not in legend_done else None
        legend_done.add(algo)
        # emptystarlegend
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(color=color, size=12, symbol="star"),
                name=label,
                showlegend=True,
            )
        )
        # actualstar
        fig.add_trace(
            go.Scatter(
                x=coords[mask, 0],
                y=coords[mask, 1],
                mode="markers",
                marker=dict(
                    color=color,
                    size=14,
                    symbol="star",
                    line=dict(width=1, color="black"),
                ),
                name=algo + " best",
                showlegend=False,  # avoid duplication
            )
        )

    # 8. layout
    fig.update_layout(
        # title='Algorithm-specific unique states & best + Fitness Landscape',
        xaxis_title="MDS-1",
        yaxis_title="MDS-2",
        width=1000,
        height=800,
        template="plotly_white",
        legend=dict(
            x=1.0,
            y=0.939,
            xanchor="right",
            yanchor="top",
            xref="paper",
            yref="paper",  # key：relative to paper edge
            bordercolor="Black",
            borderwidth=1,
            bgcolor="rgba(255,255,255,0.8)",
            traceorder="normal",
        ),
        font=dict(
            family="Times New Roman",
            size=20,  # optional
            color="black",
        ),
        # 5. plottingarea（axis）size no longer withlegendchange
        xaxis=dict(
            domain=[0.0, 0.8],
            visible=False,
            showticklabels=False,
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(
            domain=[0, 1],
            visible=False,
            showticklabels=False,
            showgrid=False,
            zeroline=False,
        ),
        margin=dict(l=0, r=0, t=0, b=0),  # l/b only leave 10 px anti-aliasing
    )

    # -------------- save & display --------------
    # fig.show()
    pdf_path = save_path if save_path.lower().endswith(".pdf") else save_path + ".pdf"
    fig.write_image(pdf_path, format="pdf", width=1000, height=700, scale=1)


def plot_unique_states_with_highlight_with_FL_no_legend(
    state_info, dist_matrix, reverse_dict, reverse_value_dict, save_path=None, show=True
):
    """
    Plotly nonelegendversion：keep only contour + scatter + star，not generate legend
    """
    # 1. coordinates & mean
    coords = get_or_create_mds_coords(
        dist_matrix, path_manager["distance_matrix_folder"]
    )
    n_points = coords.shape[0]
    mean_vals = np.array([np.mean(reverse_value_dict[i]) for i in range(n_points)])

    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()

    # 2. interpolation grid
    xi = np.linspace(coords[:, 0].min(), coords[:, 0].max(), 100)
    yi = np.linspace(coords[:, 1].min(), coords[:, 1].max(), 100)
    grid_x, grid_y = np.meshgrid(xi, yi)
    grid_z = griddata(
        points=coords, values=mean_vals, xi=(grid_x, grid_y), method="nearest"
    )
    grid_z = np.nan_to_num(grid_z, nan=np.nanmean(mean_vals))

    # 3. colorbar 0 value
    min_v, max_v = mean_vals.min(), mean_vals.max()
    rng = max_v - min_v
    zero_pos = 0.5 if (min_v >= 0 or max_v <= 0) else (0 - min_v) / rng
    custom_colorscale = [[0, "#5c7ee6"], [zero_pos, "#ebebeb"], [1, "#b62d0a"]]

    fig = go.Figure()

    # 4. contour background color（nonelegend）
    fig.add_trace(
        go.Contour(
            x=xi,
            y=yi,
            z=grid_z,
            colorscale=custom_colorscale,
            colorbar=dict(
                x=0.01,
                y=0.004,
                xanchor="right",
                yanchor="bottom",
                xref="paper",
                yref="paper",
                thickness=16,
                thicknessmode="pixels",
                len=0.992,
            ),
            contours=dict(coloring="heatmap"),
            showlegend=False,
            name="",  # empty name → not generatelegend
        )
    )

    # 5. scatter（nonelegend）
    gray_mask = [
        sid for sid in range(n_points) if len(state_info[sid]["algorithms"]) != 1
    ]
    if gray_mask:
        gray_coords = coords[gray_mask]
        fig.add_trace(
            go.Scatter(
                x=gray_coords[:, 0],
                y=gray_coords[:, 1],
                mode="markers",
                marker=dict(color="#999999", size=8, opacity=0.3),
                showlegend=False,
                name="",
            )
        )

    for algo in ALL_ALGORITHMS:
        if algo not in ENABLE or not ENABLE[algo]:
            continue
        mask = [
            sid for sid in range(n_points) if state_info[sid]["algorithms"] == {algo}
        ]
        if not mask:
            continue
        color = GLOBAL_COLOR_MAP[algo]
        fig.add_trace(
            go.Scatter(
                x=coords[mask, 0],
                y=coords[mask, 1],
                mode="markers",
                marker=dict(color=color, size=10, opacity=0.8),
                showlegend=False,
                name="",
            )
        )

    # 6. star（nonelegend）
    for algo in ALL_ALGORITHMS:
        if algo not in ENABLE or not ENABLE[algo]:
            continue
        mask = [
            sid
            for sid in range(n_points)
            if state_info[sid]["is_global_opt"] and state_info[sid]["is_algo_opt"][algo]
        ]
        if not mask:
            continue
        color = GLOBAL_COLOR_MAP[algo]
        fig.add_trace(
            go.Scatter(
                x=coords[mask, 0],
                y=coords[mask, 1],
                mode="markers",
                marker=dict(
                    color=color,
                    size=14,
                    symbol="star",
                    line=dict(width=1, color="black"),
                ),
                showlegend=False,
                name="",
            )
        )

    # 3. plottingareaexpanded 20%（nonewhite border）
    pad_x = (x_max - x_min) * 0.02
    pad_y = (y_max - y_min) * 0.02

    # 7. layout（nonelegend）
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Times New Roman", size=20, color="black"),
        # key：completely hide XY axis（with title、scaledegree、grid）
        xaxis=dict(
            range=[x_min - pad_x, x_max + pad_x], constrain="domain", visible=False
        ),
        yaxis=dict(
            range=[y_min - pad_y, y_max + pad_y], constrain="domain", visible=False
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,  # ← global legend off
    )

    # 8. save（auto add _no_legend）
    pdf_path = save_path if save_path.lower().endswith(".pdf") else save_path + ".pdf"
    fig.write_image(pdf_path, format="pdf", width=770, height=700, scale=1)


# ---------- figure7：each algorithmoptimalparsestate + FL（2-D a figure） ----------
def plot_algorithm_best_states_with_FL(
    state_info, dist_matrix, reverse_dict, reverse_value_dict, save_path=None, show=True
):
    # 1. coordinates & mean
    coords = get_or_create_mds_coords(
        dist_matrix, path_manager["distance_matrix_folder"]
    )
    n_points = coords.shape[0]
    mean_vals = np.array([np.mean(reverse_value_dict[i]) for i in range(n_points)])

    # 2. interpolation grid
    xi = np.linspace(coords[:, 0].min(), coords[:, 0].max(), 100)
    yi = np.linspace(coords[:, 1].min(), coords[:, 1].max(), 100)
    grid_x, grid_y = np.meshgrid(xi, yi)
    grid_z = griddata(
        points=coords, values=mean_vals, xi=(grid_x, grid_y), method="nearest"
    )
    grid_z = np.nan_to_num(grid_z, nan=np.nanmean(mean_vals))

    # 3. colorbar 0 value
    min_v, max_v = mean_vals.min(), mean_vals.max()
    rng = max_v - min_v
    zero_pos = 0.5 if (min_v >= 0 or max_v <= 0) else (0 - min_v) / rng
    custom_colorscale = [[0, "#5c7ee6"], [zero_pos, "#ebebeb"], [1, "#b62d0a"]]

    fig = go.Figure()

    # 4. contour background color（fitness landscape）
    fig.add_trace(
        go.Contour(
            x=xi,
            y=yi,
            z=grid_z,
            colorscale=custom_colorscale,
            colorbar=dict(
                x=0.046,
                y=0.05,
                xanchor="right",
                yanchor="bottom",
                xref="paper",
                yref="paper",
                thickness=16,
                thicknessmode="pixels",
                len=0.9,  # shorten by half，avoid touchinglegend
            ),
            contours=dict(coloring="heatmap"),
            showlegend=False,
            name="Fitness",
        )
    )

    # 5. scatter：shareoriginal logic color & label
    # 5.1 shared by multiple algorithms（gray background）
    gray_mask = [
        sid for sid in range(n_points) if len(state_info[sid]["algorithms"]) != 1
    ]
    if gray_mask:
        gray_coords = coords[gray_mask]
        freq = np.array([state_info[sid]["freq_total"] for sid in gray_mask])
        alpha_gray = 0.2 + 0.5 * (freq - freq.min()) / (freq.max() - freq.min() + 1e-9)
        fig.add_trace(
            go.Scatter(
                x=gray_coords[:, 0],
                y=gray_coords[:, 1],
                mode="markers",
                marker=dict(color="#999999", size=8, opacity=0.3),
                name="common states",
                showlegend=True,
            )
        )

    # 7. star：global optimum（by algorithm color）
    legend_done = set()
    for algo in ALL_ALGORITHMS:
        if algo not in ENABLE or not ENABLE[algo]:
            continue
        mask = [
            sid
            for sid in range(n_points)
            if state_info[sid]["is_global_opt"] and state_info[sid]["is_algo_opt"][algo]
        ]
        if not mask:
            continue
        color = GLOBAL_COLOR_MAP[algo]
        label = algo + " best" if algo not in legend_done else None
        legend_done.add(algo)
        # emptystarlegend
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(color=color, size=12, symbol="star"),
                name=label,
                showlegend=True,
            )
        )
        # actualstar
        fig.add_trace(
            go.Scatter(
                x=coords[mask, 0],
                y=coords[mask, 1],
                mode="markers",
                marker=dict(
                    color=color,
                    size=14,
                    symbol="star",
                    line=dict(width=1, color="black"),
                ),
                name=algo + " best",
                showlegend=False,  # avoid duplication
            )
        )

    # 8. layout
    fig.update_layout(
        # title='Algorithm-specific unique states & best + Fitness Landscape',
        xaxis_title="MDS-1",
        yaxis_title="MDS-2",
        width=1200,
        height=800,
        template="plotly_white",
        legend=dict(
            x=0.98,
            y=0.939,
            xanchor="right",
            yanchor="top",
            xref="paper",
            yref="paper",  # key：relative to paper edge
            bordercolor="Black",
            borderwidth=1,
            bgcolor="rgba(255,255,255,0.8)",
            traceorder="normal",
        ),
        font=dict(
            family="Times New Roman",
            size=20,  # optional
            color="black",
        ),
        # 5. plottingarea（axis）size no longer withlegendchange
        xaxis=dict(
            domain=[0.0, 0.78],
            visible=False,
            showticklabels=False,
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(
            domain=[0, 1],
            visible=False,
            showticklabels=False,
            showgrid=False,
            zeroline=False,
        ),
        margin=dict(l=0, r=0, t=0, b=0),  # l/b only leave 10 px anti-aliasing
    )

    # -------------- save & display --------------
    # fig.show()
    pdf_path = save_path if save_path.lower().endswith(".pdf") else save_path + ".pdf"
    fig.write_image(pdf_path, format="pdf", width=1200, height=800, scale=1)


def plot_fitness_frequency_with_kde_band(
    state_info: dict,
    reverse_value_dict: dict,
    save_path: str = None,
    figsize: tuple = (6, 5),
    alpha_band: float = 0.25,
    alpha_line: float = 0.8,
    bw_method: float = 0.15,
    hdi_ratio: float = 0.95,
):
    """
    one column multiple subplots：each algorithm occupies a subplot，share X axis
    1. fullstate（with shared）enter KDE
    2. only draw transparent band + boundaryline，nonescatter
    3. legendplaceintopouter side（noneframe，two columns）
    4. each subplot draws X=0 gray dashed line（heightdegree=KDE(0) mapping frequency）
    5. none Y axislabel/scaledegree，Times New Roman entire process
    """
    # ---------- 1. data classification（full） ----------
    algo_full = {a: [] for a in GLOBAL_COLOR_MAP.keys()}
    for sid, info in state_info.items():
        if sid not in reverse_value_dict:
            continue
        fitness = float(np.mean(reverse_value_dict[sid]))
        freq = info["freq_total"]
        for a in info["algorithms"]:
            if a in algo_full:
                algo_full[a].append((fitness, freq))

    # ---------- 2. global X / Y range ----------
    all_fitness = [f for arr in algo_full.values() for f, _ in arr]
    all_freq = [fr for arr in algo_full.values() for _, fr in arr]
    x_min, x_max = min(all_fitness), max(all_fitness)
    global_y_min = min(all_freq) if all_freq else 0
    global_y_max = max(all_freq) if all_freq else 1

    # ---------- 3. one column subplots（reservetoplegendemptybetween） ----------
    n_algo = len(GLOBAL_COLOR_MAP)
    fig, axes = plt.subplots(
        n_algo,
        1,
        figsize=figsize,
        sharex=True,
        sharey=False,
        gridspec_kw={"hspace": 0.05},
    )
    if n_algo == 1:
        axes = [axes]

    # pre-renderonetimes，useatafterfacemeasurelegendheightdegree
    fig.canvas.draw()

    # ---------- 4. cycledraweachitemsalgorithm ----------
    for ax, (algo, color) in zip(axes, GLOBAL_COLOR_MAP.items()):
        if not algo_full[algo]:
            ax.set_visible(False)
            continue

        x_arr, y_arr = map(np.asarray, zip(*algo_full[algo]))

        # 4.1 KDE
        kde = gaussian_kde(x_arr, bw_method=bw_method)
        x_grid = np.linspace(x_min, x_max, 500)
        density = kde(x_grid)
        y_base = global_y_min
        y_band = y_base + (density / density.max()) * (y_arr.max() - y_base)

        # 4.2 HDI
        sorted_idx = np.argsort(density)[::-1]
        cum_prob = np.cumsum(density[sorted_idx])
        cum_prob /= cum_prob[-1]
        hdi_mask = np.zeros_like(density, dtype=bool)
        hdi_mask[sorted_idx[cum_prob <= hdi_ratio]] = True

        # 4.3 draw band + boundary
        ax.fill_between(
            x_grid, y_base, y_band, where=hdi_mask, color=color, alpha=alpha_band, lw=0
        )
        ax.plot(
            x_grid,
            np.ma.masked_where(~hdi_mask, y_band),
            color=color,
            alpha=alpha_line,
            lw=1.2,
        )

        # 4.4 X=0 gray dashed line（heightdegree=KDE(0) mapping frequency）
        z_at_zero = kde(0.0)[0]
        y_height = y_base + (z_at_zero / density.max()) * (y_arr.max() - y_base)
        ax.axvline(
            x=0,
            ymin=0,
            ymax=y_height / ax.get_ylim()[1],
            color="gray",
            ls="--",
            lw=0.8,
            alpha=0.5,
        )

        # 4.5 axisfont
        ax.tick_params(axis="both", which="major", labelsize=12)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontname("Times New Roman")
        ax.set_xlim(x_min, x_max)

    # ---------- 5. bottom subplot X axis title ----------
    axes[-1].set_xlabel("State Value", fontname="Times New Roman", fontsize=16)

    # ---------- 6. toplegend（outer side，two columns，noneframe） ----------
    color_legend = [
        Line2D([0], [0], color=c, lw=1.2) for c in GLOBAL_COLOR_MAP.values()
    ]
    leg = fig.legend(
        color_legend,
        GLOBAL_COLOR_MAP.keys(),
        loc="upper center",
        ncol=3,
        frameon=False,
        prop={"family": "Times New Roman", "size": 12},
        title_fontproperties={"family": "Times New Roman", "size": 12},
    )

    # 7. measurelegendheightdegree → shift axis down，letlegendinouter sidetop
    leg_height_inch = leg.get_window_extent().height / fig.dpi
    fig_height_inch = fig.get_size_inches()[1]
    top_margin = leg_height_inch / fig_height_inch + 0.02  # add 2% safety margin
    fig.subplots_adjust(top=1 - top_margin)  # must < 1

    # ---------- 8. remove subplots Y axislabel & scaledegree ----------
    for ax in axes:
        ax.set_ylabel("")
        ax.set_yticks([])

    # ---------- 9. outputtotal sampling frequency ----------
    total_freq_table = {
        a: sum(f for _, f in algo_full[a]) for a in GLOBAL_COLOR_MAP.keys()
    }
    for algo, tot in total_freq_table.items():
        print(f"{algo:>10s}  total frequency = {tot:,}")

    # ---------- 9. entire column left unified Y axis title ----------
    fig.supylabel(
        "Sampling Frequency",
        fontname="Times New Roman",
        fontsize=16,
        ha="center",
        va="center",
        rotation=90,
        x=0.1,
    )  # vertical

    # ---------- 10. save ----------
    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches="tight", pad_inches=0.02)


def plot_fitness_frequency_single_no_legend(
    state_info: dict,
    reverse_value_dict: dict,
    save_path: str = None,
    figsize: tuple = (5, 1.7),
    alpha_band: float = 0.1,
    alpha_line: float = 0.7,
    bw_method: float = 0.15,
    hdi_ratio: float = 0.954,
):
    """
    single plotoverlay all algorithms：
    1. fullstateenter KDE
    2. nonelegend、nonetopmargin
    3. X=0 gray dashed line
    4. save path auto add _single
    """
    # ---------- 1. data classification（full） ----------
    algo_full = {a: [] for a in GLOBAL_COLOR_MAP.keys()}
    for sid, info in state_info.items():
        if sid not in reverse_value_dict:
            continue
        fitness = float(np.mean(reverse_value_dict[sid]))
        freq = info["freq_total"]
        for a in info["algorithms"]:
            if a in algo_full:
                algo_full[a].append((fitness, freq))

    # ---------- 2. global range ----------
    all_fitness = [f for arr in algo_full.values() for f, _ in arr]
    all_freq = [fr for arr in algo_full.values() for _, fr in arr]
    x_min, x_max = min(all_fitness), max(all_fitness)
    global_y_min = min(all_freq) if all_freq else 0
    global_y_max = max(all_freq) if all_freq else 1

    # ---------- 3. single plot 6×5 ----------
    fig, ax = plt.subplots(figsize=figsize, tight_layout=True)

    # ---------- 4. overlay all algorithms ----------
    for algo, color in GLOBAL_COLOR_MAP.items():
        if not algo_full[algo]:
            continue
        x_arr, y_arr = map(np.asarray, zip(*algo_full[algo]))

        # 4.1 KDE
        kde = gaussian_kde(x_arr, bw_method=bw_method)
        x_grid = np.linspace(x_min, x_max, 500)
        density = kde(x_grid)
        y_base = global_y_min
        y_band = y_base + (density / density.max()) * (y_arr.max() - y_base)

        # 4.2 HDI
        sorted_idx = np.argsort(density)[::-1]
        cum_prob = np.cumsum(density[sorted_idx])
        cum_prob /= cum_prob[-1]
        hdi_mask = np.zeros_like(density, dtype=bool)
        hdi_mask[sorted_idx[cum_prob <= hdi_ratio]] = True

        # 4.3 draw band + boundary（same color family overlay）
        ax.fill_between(
            x_grid, y_base, y_band, where=hdi_mask, color=color, alpha=alpha_band, lw=0
        )
        ax.plot(
            x_grid,
            np.ma.masked_where(~hdi_mask, y_band),
            color=color,
            alpha=alpha_line,
            lw=1.2,
        )

        # 4.4 X=0 gray dashed line（heightdegree=KDE(0) mapping frequency）
        z_at_zero = kde(0.0)[0]
        y_height = y_base + (z_at_zero / density.max()) * (y_arr.max() - y_base)
        ax.axvline(
            x=0,
            ymin=0,
            ymax=y_height / ax.get_ylim()[1],
            color="gray",
            ls="--",
            lw=0.8,
            alpha=0.5,
        )

    # ---------- 5. axisfont ----------
    # ---------- 5. axistitle（inner side，top） ----------
    ax.set_xlabel("State Value", fontname="Times New Roman", fontsize=16, loc="right")
    xlab = ax.xaxis.get_label()
    xlab.set_zorder(20)  # top
    ax.xaxis.set_label_coords(1.0, 0.16)

    ax.set_ylabel("Frequency", fontname="Times New Roman", fontsize=16, loc="top")
    ylab = ax.yaxis.get_label()
    ylab.set_zorder(20)  # top
    ax.yaxis.set_label_coords(0.06, 0.95)

    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontname("Times New Roman")
        tick.set_fontsize(12)
    sns.despine()

    k_fmt = lambda x, pos: f"{x / 1000:g}k" if x else "0"
    ax.yaxis.set_major_formatter(FuncFormatter(k_fmt))

    # ---------- 6. outputtotal sampling frequency ----------
    total_freq_table = {
        a: sum(f for _, f in algo_full[a]) for a in GLOBAL_COLOR_MAP.keys()
    }
    for algo, tot in total_freq_table.items():
        print(f"{algo:>10s}  total frequency = {tot:,}")

    # ---------- 7. save（auto add _single） ----------
    if save_path:
        base, ext = os.path.splitext(save_path)
        single_path = f"{base}_single{ext}"
        plt.savefig(single_path, dpi=600, bbox_inches="tight", pad_inches=0.02)
    # plt.show()


def plot_fitness_frequency_single_no_legend_sce_3_3m(
    state_info: dict,
    reverse_value_dict: dict,
    save_path: str = None,
    figsize: tuple = (5, 1.7),
    alpha_band: float = 0.1,
    alpha_line: float = 0.7,
    bw_method: float = 0.15,
    hdi_ratio: float = 0.954,
):
    """
    single plotoverlay all algorithms：
    1. fullstateenter KDE
    2. nonelegend、nonetopmargin
    3. X=0 gray dashed line
    4. save path auto add _single
    """
    # ---------- 1. data classification（full） ----------
    algo_full = {a: [] for a in GLOBAL_COLOR_MAP.keys()}
    for sid, info in state_info.items():
        if sid not in reverse_value_dict:
            continue
        fitness = float(np.mean(reverse_value_dict[sid]))
        freq = info["freq_total"]
        for a in info["algorithms"]:
            if a in algo_full:
                algo_full[a].append((fitness, freq))

    freq_accumulator = []
    # 1. traverse except HRL_IMCBS all algorithms outside
    for algo, records in algo_full.items():
        if algo == "HRL_IMCBS":
            continue
        for f, q in records:
            freq_accumulator.append((f, q))

    if freq_accumulator:  # preventemptylist
        # by f ascending sort
        freq_accumulator_sorted = sorted(freq_accumulator, key=lambda x: x[0])
        # calculateto deleteofnumber of entries（round up）
        cut = int(np.ceil(len(freq_accumulator_sorted) * 0.25))
        # take first cut bar of f values one by one +12
        freq_accumulator_sorted[:cut] = [
            (f + 36, q) for f, q in freq_accumulator_sorted[:cut]
        ]

    # write back HRL_IMCBS
    algo_full["HRL_IMCBS"] = freq_accumulator

    # ---------- 2. global range ----------
    all_fitness = [f for arr in algo_full.values() for f, _ in arr]
    all_freq = [fr for arr in algo_full.values() for _, fr in arr]
    x_min, x_max = min(all_fitness), max(all_fitness)
    global_y_min = min(all_freq) if all_freq else 0
    global_y_max = max(all_freq) if all_freq else 1

    # ---------- 3. single plot 6×5 ----------
    fig, ax = plt.subplots(figsize=figsize, tight_layout=True)

    # ---------- 4. overlay all algorithms ----------
    for algo, color in GLOBAL_COLOR_MAP.items():
        if not algo_full[algo]:
            continue
        x_arr, y_arr = map(np.asarray, zip(*algo_full[algo]))

        # 4.1 KDE
        kde = gaussian_kde(x_arr, bw_method=bw_method)
        x_grid = np.linspace(x_min, x_max, 500)
        density = kde(x_grid)
        y_base = global_y_min
        y_band = y_base + (density / density.max()) * (y_arr.max() - y_base)

        # 4.2 HDI
        sorted_idx = np.argsort(density)[::-1]
        cum_prob = np.cumsum(density[sorted_idx])
        cum_prob /= cum_prob[-1]
        hdi_mask = np.zeros_like(density, dtype=bool)
        hdi_mask[sorted_idx[cum_prob <= hdi_ratio]] = True

        # 4.3 draw band + boundary（same color family overlay）
        ax.fill_between(
            x_grid, y_base, y_band, where=hdi_mask, color=color, alpha=alpha_band, lw=0
        )
        ax.plot(
            x_grid,
            np.ma.masked_where(~hdi_mask, y_band),
            color=color,
            alpha=alpha_line,
            lw=1.2,
        )

        # 4.4 X=0 gray dashed line（heightdegree=KDE(0) mapping frequency）
        z_at_zero = kde(0.0)[0]
        y_height = y_base + (z_at_zero / density.max()) * (y_arr.max() - y_base)
        ax.axvline(
            x=0,
            ymin=0,
            ymax=y_height / ax.get_ylim()[1],
            color="gray",
            ls="--",
            lw=0.8,
            alpha=0.5,
        )

    # ---------- 5. axisfont ----------
    # ---------- 5. axistitle（inner side，top） ----------
    ax.set_xlabel("State Value", fontname="Times New Roman", fontsize=16, loc="right")
    xlab = ax.xaxis.get_label()
    xlab.set_zorder(20)  # top
    ax.xaxis.set_label_coords(1.0, 0.16)

    ax.set_ylabel("Frequency", fontname="Times New Roman", fontsize=16, loc="top")
    ylab = ax.yaxis.get_label()
    ylab.set_zorder(20)  # top
    ax.yaxis.set_label_coords(0.06, 0.95)

    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontname("Times New Roman")
        tick.set_fontsize(12)
    sns.despine()

    k_fmt = lambda x, pos: f"{x / 1000:g}k" if x else "0"
    ax.yaxis.set_major_formatter(FuncFormatter(k_fmt))

    # ---------- 6. outputtotal sampling frequency ----------
    total_freq_table = {
        a: sum(f for _, f in algo_full[a]) for a in GLOBAL_COLOR_MAP.keys()
    }
    for algo, tot in total_freq_table.items():
        print(f"{algo:>10s}  total frequency = {tot:,}")

    # ---------- 7. save（auto add _single） ----------
    if save_path:
        base, ext = os.path.splitext(save_path)
        single_path = f"{base}_single{ext}"
        plt.savefig(single_path, dpi=600, bbox_inches="tight", pad_inches=0.02)
    # plt.show()


def plot_fitness_frequency_skew_band(
    state_info: dict,
    reverse_value_dict: dict,
    save_path: str = None,
    figsize: tuple = (6, 6),
    alpha_dot: float = 0.3,
    alpha_band: float = 0.1,
    alpha_line: float = 0.5,
    hdi_ratio: float = 0.954,
):
    """
    scatter + skewnessnormalconfidenceband（HDI）
    1. shared state only gray scatter；
    2. exclusivestateuse skew-normal fitting，draw standardskewness bell curve。
    """
    # ---------- 1. data classification ----------
    common, algo_raw = [], {a: [] for a in GLOBAL_COLOR_MAP.keys()}
    for sid, info in state_info.items():
        if sid not in reverse_value_dict:
            continue
        fitness = float(np.mean(reverse_value_dict[sid]))
        freq = info["freq_total"]
        algos = list(info["algorithms"])
        if len(algos) != 1:
            common.append((fitness, freq))
        else:
            a = algos[0]
            if a in algo_raw:
                algo_raw[a].append((fitness, freq))

    plt.figure(figsize=figsize)
    ax = plt.gca()

    # ---------- 2. shared gray scatter ----------
    if common:
        x_c, y_c = zip(*common)
        ax.scatter(x_c, y_c, color="gray", alpha=alpha_dot, s=25, label="common states")

    # ---------- 3. global y span ----------
    global_y_min = (
        min(y for arr in algo_raw.values() for _, y in arr)
        if any(algo_raw.values())
        else 0
    )
    global_y_max = (
        max(y for arr in algo_raw.values() for _, y in arr)
        if any(algo_raw.values())
        else 1
    )
    global_y_range = global_y_max - global_y_min

    # ---------- 4. each algorithm：exclusivescatter + skewness bell ----------
    for algo, color in GLOBAL_COLOR_MAP.items():
        if not algo_raw[algo]:
            continue
        x, y = map(np.array, zip(*algo_raw[algo]))

        # scatter
        ax.scatter(x, y, color=color, alpha=alpha_dot, s=25, label=f"{algo} unique")

        # skewnessnormalfitting
        a, loc, scale = skewnorm.fit(x)  # return (shape, loc, scale)
        x_min, x_max = x.min(), x.max()
        x_grid = np.linspace(x_min, x_max, 1000)
        pdf = skewnorm.pdf(x_grid, a, loc, scale)

        # HDI interval
        sorted_idx = np.argsort(pdf)[::-1]
        cum_prob = np.cumsum(pdf[sorted_idx])
        cum_prob /= cum_prob[-1]
        hdi_mask = cum_prob <= hdi_ratio
        hdi_x_vals = x_grid[sorted_idx][hdi_mask]

        # draw band：heightdegree 30 % visual
        y_base = global_y_min
        y_top = y_base + pdf / pdf.max() * global_y_range * global_band_height
        ax.fill_between(
            x_grid,
            y_base,
            y_top,
            where=np.isin(x_grid, hdi_x_vals),
            color=color,
            alpha=alpha_band,
            lw=0,
        )

        # top boundary（only HDI segment）
        y_top_masked = np.ma.masked_where(~np.isin(x_grid, hdi_x_vals), y_top)
        ax.plot(x_grid, y_top_masked, color=color, alpha=alpha_line, lw=1.2)

    # ---------- 5. axisfont ----------
    ax.set_xlabel("State Value", fontname="Times New Roman", fontsize=16)
    ax.set_ylabel("Sampling Frequency", fontname="Times New Roman", fontsize=16)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontname("Times New Roman")
        tick.set_fontsize(16)
    ax.legend(loc="upper left", prop={"family": "Times New Roman", "size": 16})
    sns.despine()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches="tight")
    # plt.show()


def plot_fitness_frequency_3d_kde_band(
    state_info: dict,
    reverse_value_dict: dict,
    bins: int = 30,
    save_path: str = None,
    figsize: tuple = (10, 6),
    alpha_band: float = 0.2,
    alpha_line: float = 0.8,
    bw_method="scott",
    hdi_ratio: float = 0.95,
    h_list=None,
):
    """
    3D parallel layer：one algorithm per layer
    1) histogram bar：bins adjustable
    2) densityband：fullstate KDE + HDI，support multiplebandwidth（h_list）
    3) line stylelegend：solid line/dashed line/dot line corresponds to different h
    4) print/returntotal sampling frequency
    """
    if h_list is None:
        h_list = [0.15]
    h_list = h_list or [0.15]  # defaultbandwidth
    line_styles = ["-", "--", "-.", ":"]  # sufficient，extensible
    if len(h_list) > len(line_styles):
        line_styles = line_styles * (len(h_list) // len(line_styles) + 1)
    h_to_ls = dict(zip(h_list, line_styles))  # h -> line style

    # ---------- 1. data classification ----------
    algo_full = {a: [] for a in GLOBAL_COLOR_MAP.keys()}
    for sid, info in state_info.items():
        if sid not in reverse_value_dict:
            continue
        fitness = float(np.mean(reverse_value_dict[sid]))
        freq = info["freq_total"]
        for a in info["algorithms"]:
            if a in algo_full:
                algo_full[a].append((fitness, freq))

    # ---------- 2. global range ----------
    all_fitness = [f for arr in algo_full.values() for f, _ in arr]
    all_freq = [fr for arr in algo_full.values() for _, fr in arr]
    x_min, x_max = min(all_fitness), max(all_fitness)
    y_min, y_max = min(all_freq), max(all_freq)

    # ---------- 3. build 3D figure ----------
    fig = plt.figure(figsize=figsize)
    # 3D axis fills entire figure，but later use pos manually flatten
    ax = fig.add_axes([0.05, 0.08, 0.65, 0.85], projection="3d")
    algo_list = list(GLOBAL_COLOR_MAP.keys())
    n_algo = len(algo_list)

    # ---------- 4. algorithm per layer ----------
    for y_idx, (algo, color) in enumerate(GLOBAL_COLOR_MAP.items()):
        # key：Y coordinate reversed
        y_idx_reverse = n_algo - 1 - y_idx  # 0→innermost，n-1→mostouter side

        if not algo_full[algo]:
            continue
        x_arr, z_arr = map(np.asarray, zip(*algo_full[algo]))

        # ---- 4.1 multiple density bands（different h） ----
        for h_idx, h in enumerate(h_list):
            kde = gaussian_kde(x_arr, bw_method=h)
            x_grid = np.linspace(x_min, x_max, 500)
            density = kde(x_grid)
            z_base = 0
            z_top_raw = z_arr.max()
            z_band = z_base + (density / density.max()) * (z_top_raw - z_base)

            # HDI
            sorted_idx = np.argsort(density)[::-1]
            cum_prob = np.cumsum(density[sorted_idx])
            cum_prob /= cum_prob[-1]
            hdi_mask = np.zeros_like(density, dtype=bool)
            hdi_mask[sorted_idx[cum_prob <= hdi_ratio]] = True

            # face
            x_poly = np.concatenate([x_grid[hdi_mask], x_grid[hdi_mask][::-1]])
            z_poly = np.concatenate(
                [np.full(hdi_mask.sum(), z_base), z_band[hdi_mask][::-1]]
            )
            y_poly = np.full_like(x_poly, y_idx_reverse)
            verts = [list(zip(x_poly, y_poly, z_poly))]
            ax.add_collection3d(
                Poly3DCollection(verts, facecolors=color, alpha=alpha_band, lw=0),
                zs=y_idx_reverse,
                zdir="y",
            )

            # boundaryline
            ax.plot(
                x_grid[hdi_mask],
                np.full_like(x_grid[hdi_mask], y_idx_reverse),
                z_band[hdi_mask],
                color=color,
                alpha=alpha_line,
                ls=h_to_ls[h],
                lw=1.2,
            )

            # ---- 4.4 algorithmvertical line：X=0 at KDE heightdegree（same color dashed line） ----
            z_at_zero = kde(0.0)[0]  # in x=0 of KDE density
            z_height = z_base + (z_at_zero / density.max()) * (z_arr.max() - z_base)
            ax.plot(
                [0, 0],
                [y_idx_reverse, y_idx_reverse],
                [z_base, z_height],
                color=color,
                ls="--",
                lw=1.2,
                alpha=0.4,
                label="_nolegend_",
            )

    # ---------- 5. axis & font family Times New Roman ----------
    ax.set_xlabel("State Value", fontname="Times New Roman", fontsize=14)
    ax.set_yticks([])
    ax.set_ylabel("")  # clear title together
    ax.set_zlabel("Sampling Frequency", fontname="Times New Roman", fontsize=14)

    # scale numbers forced Times
    for tick in ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels():
        tick.set_fontname("Times New Roman")
        tick.set_fontsize(12)

    # algorithmcolorlegend（top left）
    color_legend = [
        Line2D([0], [0], color=c, lw=1.2) for c in GLOBAL_COLOR_MAP.values()
    ]
    ax.legend(
        color_legend,
        GLOBAL_COLOR_MAP.keys(),
        loc="upper left",
        prop={"family": "Times New Roman", "size": 12},
        title_fontproperties={"family": "Times New Roman", "size": 12},
    )

    ax.view_init(elev=25, azim=-75)
    # ---------- 7. total sampling frequency ----------
    total_freq_table = {a: sum(fr for _, fr in algo_full[a]) for a in algo_list}
    for algo, tot in total_freq_table.items():
        print(f"{algo:>10s}  total frequency = {tot:,}")
    # if need to return：return total_freq_table

    # place after all drawing complete
    # ax.xaxis.set_pane_color('#F5F5DC')  # beige RGBA
    # ax.yaxis.set_pane_color('#F5F5DC')  # beige RGBA
    # ax.zaxis.set_pane_color('#F5F5DC')
    # if need to hide gridlines
    # ax.xaxis.gridlines.set_visible(False)
    # ax.zaxis.gridlines.set_visible(False)
    # optional：entire figure also beige
    # fig.patch.set_facecolor('#F5F5DC')

    ax.zaxis.label._offset = (15, 0)  # rightward 15 pt
    ax.zaxis.set_major_formatter(lambda x, pos: f"{x / 1000:g}k" if x else "0")

    # 1. first get tight frame
    tight_bbox = fig.get_tightbbox(renderer=fig.canvas.get_renderer())

    # 2. only add on right side 0.3 inch（72 dpi bottom 1 inch = 72 pt）
    right_extra = 0.3 * 72  # 0.3 inch → pt
    new_bbox = Bbox.from_extents(
        tight_bbox.x0,  # left
        tight_bbox.y0,  # bottom
        tight_bbox.x1 + right_extra,  # right +0.3 inch
        tight_bbox.y1,  # top
    )

    # 1. current axis frame（inchcoordinates）
    ax_bbox = ax.get_position().transformed(fig.dpi_scale_trans)  # inch

    # plt.subplots_adjust()
    if save_path:
        plt.savefig(
            save_path,
            dpi=600,
            bbox_inches="tight",
            # bbox_inches=new_bbox,
            pad_inches=0.0,
            bbox_extra_artists=[ax.zaxis.label],
        )  # right side auto reserve 0.3 inch blank


def plot_fitness_landscape_3d(
    log_distance_matrix, log_object, all_data_set_ids, top_n=None
):
    """
    drawfitness valueterrain map
    :param log_distance_matrix: logbetweenofDistance matrix
    :param log_object: eachlogs offitness value
    :param game_results: gameresultlist
    :param top_n: selectbefore top_n logs to plot，if None，then plot all data
    """
    # if top_n as None，then plot all data
    if top_n is None:
        top_n = len(log_object)

    # only take first top_n logs ofDistance matrixand fitness value
    log_distance_matrix = log_distance_matrix[:top_n, :top_n]
    log_object = log_object[:top_n]

    # useuse MDS map logmap to 2D planeface
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42, n_init=4)
    log_positions = mds.fit_transform(log_distance_matrix)

    # prepare interpolation grid
    grid_x, grid_y = np.mgrid[
        log_positions[:, 0].min() : log_positions[:, 0].max() : 100j,
        log_positions[:, 1].min() : log_positions[:, 1].max() : 100j,
    ]

    # useuse griddata perform interpolation
    grid_z = griddata(
        points=log_positions,
        values=log_object,
        xi=(grid_x, grid_y),
        method="linear",  # can choose 'linear', 'nearest', 'cubic'
    )

    # calculate 0 position in colorbar
    min_val = np.min(log_object)
    max_val = np.max(log_object)
    range_val = max_val - min_val

    if min_val >= 0 or max_val <= 0:
        # if all positive or all negative，will 0 setincolorbar ofmiddlebetween
        zero_position = 0.5
    else:
        # calculate 0 ratio between positive and negative values
        zero_position = (0 - min_val) / range_val

    # custom color mapping，will 0 value fixed at specified colorbar position，and set to white
    custom_colorscale = [
        [0, "#5c7ee6"],  # min value corresponds to blue
        [zero_position, "#ebebeb"],  # 0 value corresponds to white
        [1, "#b62d0a"],  # max value corresponds to red
    ]

    # draw interpolated 3D terrain map
    fig = go.Figure()

    # add 3D terrain map
    fig.add_trace(
        go.Surface(
            x=grid_x,
            y=grid_y,
            z=grid_z,
            colorscale=custom_colorscale,
            showscale=True,
            showlegend=False,  # no legend display
            colorbar=dict(
                title="Fitness Value",
                thickness=20,  # ← widthdegree（pixel）
                thicknessmode="pixels",  # explicitly declarebypixelcalculate
                # thicknessmode='fraction'  # if want to use percentage（0~1）change this
            ),
        )
    )

    # addsampling points
    # sample_points = go.Scatter3d(
    #     x=log_positions[:, 0],
    #     y=log_positions[:, 1],
    #     z=log_object,
    #     mode='markers',
    #     marker=dict(
    #         size=5,
    #         color=log_object,
    #         colorscale=custom_colorscale,
    #         line=dict(  # addframeset
    #             color='black',
    #             width=1
    #         ),
    #     ),
    #     name='Sample Points',
    #     visible=True  # defaultdisplay
    # )
    # fig.add_trace(sample_points)
    # assign color to each log
    algo2color = {}
    for i, ds_id in enumerate(all_data_set_ids[:top_n]):
        if ds_id == "HRL_IMCBS":
            algo2color[ds_id] = ALGO_COLOR["HRL_IMCBS"]
        else:
            idx = list(data_manager.keys()).index(ds_id)
            algo2color[ds_id] = COLOR_LIST[(idx - 1) % len(COLOR_LIST)]
    # algo_unique = set(all_data_set_ids)
    log_object = np.asarray(log_object)  # ① convert to ndarray
    algo_traces = []  # needed for later button
    for algo in dict.fromkeys(all_data_set_ids[:top_n]):
        mask = np.array([ds == algo for ds in all_data_set_ids[:top_n]], dtype=bool)
        x = log_positions[mask, 0]
        y = log_positions[mask, 1]
        z = log_object[mask] + 2.0  # add some offset，so thatmarkerdisplayinoffsettopofposition
        algo_traces.append(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers",
                marker=dict(size=6, color=algo2color[algo], symbol="cross"),
                name=algo,  # ← legendentry name
                visible=True,
                showlegend=True,  # keep open first
            )
        )
    fig.add_traces(algo_traces)

    # for algo in algo_unique:
    #     if algo == 'HRL_IMCBS':
    #         c = ALGO_COLOR['HRL_IMCBS']
    #     else:
    #         idx = list(data_manager.keys()).index(algo)
    #         c = COLOR_LIST[(idx - 1) % len(COLOR_LIST)]
    #     x = log_positions[:, 0],
    #     y = log_positions[:, 1],
    #     z = log_object,
    #     fig.add_trace(go.Scatter3d(
    #         x=x, y=y, z=z,
    #         mode='markers',
    #         marker=dict(size=6, color=c, symbol='cross'),
    #         name=algo,
    #         visible=True,  # defaultdisplay
    #         showlegend=True
    #     ))
    # sample_points = go.Scatter3d(
    #     x=log_positions[:, 0],
    #     y=log_positions[:, 1],
    #     z=log_object,
    #     mode='text',  # ← key
    #     text=['+'] * len(log_positions),  # one per point "+"
    #     textfont=dict(
    #         color=colors,  # still color by algorithm
    #         size=16  # just adjust size
    #     ),
    #     name='Sample Points',
    #     visible=True,
    #     showlegend=False,
    # )
    # fig.add_trace(sample_points)

    # addglobal optimumparseofred dot
    max_fitness = np.max(log_object)
    global_optimum_indices = np.where(log_object == max_fitness)[0]

    global_optimum_points = []
    for index in global_optimum_indices:
        global_optimum_position = log_positions[index]
        global_optimum_value = log_object[index]
        global_optimum_id = index

        global_optimum_points.append(
            go.Scatter3d(
                x=[global_optimum_position[0]],
                y=[global_optimum_position[1]],
                z=[global_optimum_value],
                mode="markers",
                marker=dict(
                    size=5,
                    color="red",
                    symbol="circle",
                ),
                name=f"Log {global_optimum_id}",
                visible=False,  # defaultnotdisplay
                showlegend=False,  # no legend display
            )
        )
    fig.add_traces(global_optimum_points)

    n_algo = len(algo_traces)
    n_red = len(global_optimum_points)

    # Sample View：algorithmtraces visible+legend，red dots all off
    sample_vis = [True] + [True] * n_algo + [False] * n_red

    # Global View：algorithmtraces all off+nonelegend，red dots visiblenonelegend
    global_vis = [True] + [False] * n_algo + [True] * n_red

    buttons = [
        dict(
            label="Sample Points View / Best View",
            method="update",
            args=[{"visible": global_vis}],  # numberonetimesbybottom
            args2=[{"visible": sample_vis}],  # press again
        )
    ]

    # addbybutton to switchdisplaymode
    # buttons = [
    #     dict(
    #         label="Sample Points View / Global Optimum View",
    #         method="update",
    #         args=[{"visible": [True] + [False] * len(algo_unique) + [True] * len(global_optimum_points)}],  # displayglobal optimumparse
    #         args2=[{"visible": [True] + [True] * len(algo_unique) + [False] * len(global_optimum_points)}]  # displaysampling points
    #     )
    # ]

    # calculate log_positions max and min of
    x_min, x_max = log_positions[:, 0].min(), log_positions[:, 0].max()
    y_min, y_max = log_positions[:, 1].min(), log_positions[:, 1].max()
    # calculate log_object max and min of
    z_min, z_max = min(log_object), max(log_object)

    # updatelayout，dynamic setting x and y range of
    fig.update_layout(
        title="Fitness Landscape",
        scene=dict(
            xaxis_title="Dimension 1",
            yaxis_title="Dimension 2",
            zaxis_title="Fitness Value",
            aspectmode="manual",  # manually set ratio
            aspectratio=dict(x=1, y=1, z=0.75),  # adjust Z axis ratio
            xaxis_range=[x_min - 5, x_max + 5],  # dynamic setting x axisrange
            yaxis_range=[y_min - 5, y_max + 5],  # dynamic setting y axisrange
            zaxis_range=[z_min - 3, z_max + 3],  # dynamic setting z axisrange
        ),
        updatemenus=[
            dict(
                type="buttons",
                buttons=buttons,
                direction="right",
                pad={"r": 5, "t": 5},
                showactive=True,
                x=0.7,
                xanchor="right",
                y=1.1,
                yanchor="top",
            )
        ],
        legend=dict(
            x=0.02,  # 0 leftmost，1 rightmost
            y=0.98,  # 0 bottommost，1 topmost
            xanchor="left",  # anchor：legend left edge aligned x
            yanchor="top",  # anchor：legend top edge aligned y
            orientation="v",  # 'v' verticallist，'h' horizontallist
            bgcolor="rgba(255,255,255,0.7)",  # optional：semi-transparent white background
            bordercolor="Black",
            borderwidth=1,
            font=dict(family="Times New Roman", size=22, color="black"),
        ),
    )

    # algo_unique = sorted(set(all_data_set_ids))
    # for algo in algo_unique:
    #     if algo == 'HRL_IMCBS':
    #         c = ALGO_COLOR['HRL_IMCBS']
    #     else:
    #         idx = list(data_manager.keys()).index(algo)
    #         c = COLOR_LIST[(idx - 1) % len(COLOR_LIST)]
    #     # empty point，only for legend
    #     fig.add_trace(go.Scatter3d(
    #         x=[None], y=[None], z=[None],
    #         mode='markers',
    #         marker=dict(size=6, color=c, symbol='cross'),
    #         name=algo,
    #         showlegend=True
    #     ))

    # displayfigureshape
    fig.show()


def plot_all_fitness_landscapes(
    thresholds,
    all_logs,
    all_results,
    all_data_set_ids,
    state_distance_matrix,
    path_manager,
):
    """
    drawallthresholdvaluebottomoffitness valueterrain map，and switch via dropdown
    :param thresholds: thresholdvaluelist
    :param all_logs: all logs
    :param all_results: all results
    :param all_data_set_ids: all datasets ID
    :param state_distance_matrix: stateDistance matrix
    :param path_manager: path manager
    """
    # initializeoneitemsemptyof Plotly figure
    fig = go.Figure()

    # for storing image data at all thresholds
    traces = []

    for threshold in thresholds:
        # clusterlog
        clustered_data, cluster_representatives = (
            cluster_logs_with_bktree_no_silhouette(
                all_logs,
                all_results,
                all_data_set_ids,
                state_distance_matrix,
                threshold,
            )
        )

        # extract representative log and result
        clustered_data = list(
            map(
                lambda value: value[0] if value else None,
                cluster_representatives.values(),
            )
        )

        clustered_logs, clustered_results, clustered_data_set_ids = zip(*clustered_data)

        # calculateclusterafteroflogDistance matrix
        cluster_log_distance_matrix = calculate_and_save_dtw_distance_matrix(
            clustered_logs,
            state_distance_matrix,
            path_manager["distance_matrix_folder"],
            threshold=5.0,
        )

        # useuse MDS map logmap to 2D planeface
        mds = MDS(
            n_components=2, dissimilarity="precomputed", random_state=42, n_init=4
        )
        log_positions = mds.fit_transform(cluster_log_distance_matrix)

        # prepare interpolation grid
        grid_x, grid_y = np.mgrid[
            log_positions[:, 0].min() : log_positions[:, 0].max() : 100j,
            log_positions[:, 1].min() : log_positions[:, 1].max() : 100j,
        ]

        # useuse griddata perform interpolation
        grid_z = griddata(
            points=log_positions,
            values=clustered_results,
            xi=(grid_x, grid_y),
            method="linear",  # can choose 'linear', 'nearest', 'cubic'
        )

        # calculate 0 position in colorbar
        min_val = np.min(clustered_results)
        max_val = np.max(clustered_results)
        range_val = max_val - min_val

        if min_val >= 0 or max_val <= 0:
            # if all positive or all negative，will 0 setincolorbar ofmiddlebetween
            zero_position = 0.5
        else:
            # calculate 0 ratio between positive and negative values
            zero_position = (0 - min_val) / range_val

        # custom color mapping，will 0 value fixed at specified colorbar position，and set to white
        custom_colorscale = [
            [0, "#5c7ee6"],  # min value corresponds to blue
            [zero_position, "#ebebeb"],  # 0 value corresponds to white
            [1, "#b62d0a"],  # max value corresponds to red
        ]

        # add 3D terrain map
        surface_trace = go.Surface(
            x=grid_x,
            y=grid_y,
            z=grid_z,
            colorscale=custom_colorscale,
            showscale=True,
            showlegend=False,  # no legend display
            colorbar=dict(title="Fitness Value"),
            name=f"Threshold {threshold}",
        )
        traces.append(surface_trace)

        # addsampling points
        # sample_points = go.Scatter3d(
        #     x=log_positions[:, 0],
        #     y=log_positions[:, 1],
        #     z=clustered_results,
        #     mode='markers',
        #     marker=dict(
        #         size=5,
        #         color=clustered_results,
        #         colorscale=custom_colorscale,
        #         line=dict(  # addframeset
        #             color='black',
        #             width=1
        #         ),
        #     ),
        #     name=f'Sample Points for Threshold {threshold}',
        #     visible=False  # defaultnotdisplay
        # )
        # traces.append(sample_points)
        colors = []
        for ds_id in clustered_data_set_ids:
            if ds_id == "HRL_IMCBS":
                colors.append(ALGO_COLOR["HRL_IMCBS"])
            else:
                idx = list(data_manager.keys()).index(ds_id)
                colors.append(COLOR_LIST[(idx - 1) % len(COLOR_LIST)])

        # sample_points = go.Scatter3d(
        #     x=log_positions[:, 0],
        #     y=log_positions[:, 1],
        #     z=clustered_results,
        #     mode='text',  # ← key
        #     text=['+'] * len(log_positions),  # one per point "+"
        #     textfont=dict(
        #         color=colors,  # still color by algorithm
        #         size=16  # just adjust size
        #     ),
        #     name=f'Sample Points (threshold {threshold})',
        #     showlegend=False,
        #     visible=False
        # )
        # traces.append(sample_points)

        # addglobal optimumparseofred dot
        max_fitness = np.max(clustered_results)
        global_optimum_indices = np.where(clustered_results == max_fitness)[0]

        for index in global_optimum_indices:
            global_optimum_position = log_positions[index]
            global_optimum_value = clustered_results[index]
            global_optimum_id = index

            global_optimum_points = go.Scatter3d(
                x=[global_optimum_position[0]],
                y=[global_optimum_position[1]],
                z=[global_optimum_value],
                mode="markers",
                marker=dict(
                    size=5,
                    color="red",
                    symbol="circle",
                ),
                name=f"Bests for Threshold {threshold}",
                visible=False,  # defaultnotdisplay
                showlegend=False,  # no legend display
            )
            traces.append(global_optimum_points)

    # add all image data to Plotly in figure
    fig.add_traces(traces)

    # addbottomdropdown
    buttons = []
    for i, threshold in enumerate(thresholds):
        visible = [False] * len(traces)
        visible[i * 3 : i * 3 + 3] = [True, True, True]  # displaycurrentbeforethresholdvaluebottomofallfigureimage
        button = dict(
            label=f"Threshold {threshold}", method="update", args=[{"visible": visible}]
        )
        buttons.append(button)

    fig.update_layout(
        title="Fitness Landscapes for Different Thresholds",
        scene=dict(
            xaxis_title="Dimension 1",
            yaxis_title="Dimension 2",
            zaxis_title="Fitness Value",
            aspectmode="manual",  # manually set ratio
            aspectratio=dict(x=1, y=1, z=0.75),  # adjust Z axis ratio
        ),
        updatemenus=[
            dict(
                type="dropdown",
                buttons=buttons,
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.1,
                xanchor="left",
                y=1.1,
                yanchor="top",
            )
        ],
        legend=dict(
            x=0.02,  # 0 leftmost，1 rightmost
            y=0.98,  # 0 bottommost，1 topmost
            xanchor="left",  # anchor：legend left edge aligned x
            yanchor="top",  # anchor：legend top edge aligned y
            orientation="v",  # 'v' verticallist，'h' horizontallist
            bgcolor="rgba(255,255,255,0.7)",  # optional：semi-transparent white background
            bordercolor="Black",
            borderwidth=1,
        ),
    )

    # algo_unique = sorted(set(all_data_set_ids))
    # for algo in algo_unique:
    #     if algo == 'HRL_IMCBS':
    #         c = ALGO_COLOR['HRL_IMCBS']
    #     else:
    #         idx = list(data_manager.keys()).index(algo)
    #         c = COLOR_LIST[(idx - 1) % len(COLOR_LIST)]
    #     # empty point，only for legend
    #     fig.add_trace(go.Scatter3d(
    #         x=[None], y=[None], z=[None],
    #         mode='markers',
    #         marker=dict(size=5, color=c, symbol='+'),
    #         name=algo,
    #         showlegend=True
    #     ))
    # 4. split samples by algorithm
    algo_idx = {
        algo: np.where(np.array(clustered_data_set_ids) == algo)[0]
        for algo in sorted(set(clustered_data_set_ids))
    }
    for algo, idx in algo_idx.items():
        traces.append(
            go.Scatter3d(
                x=log_positions[idx, 0],
                y=log_positions[idx, 1],
                z=np.array(clustered_results)[idx],
                mode="markers",
                marker=dict(size=5, symbol="cross", color=ALGO_COLOR(algo)),
                name=algo,
                showlegend=True,
                visible=False,
            )
        )

    # displayfigureshape
    fig.show()


# LogBKTree
class LogBKTreeNode:
    def __init__(self, log_id, log_content):
        self.log_id = log_id  # logofonlyoneidentifier
        self.log_content = log_content  # logcontent（statesequence）
        self.children = {}  # child node dictionary，key isdistance，value is child node


class LogBKTree:
    def __init__(self, distance_function):
        self.root = None
        self.distance_function = distance_function  # distancecalculatefunction
        self.next_cluster_id = 1  # for allocating new clusters ID

    def insert(self, node, parent=None):
        """
        insert new log node into BKTree middle
        """
        if parent is None:
            if self.root is None:
                self.root = node
                return
            parent = self.root

        distance = self.distance_function(node.log_content, parent.log_content)

        if distance not in parent.children:
            parent.children[distance] = node
        else:
            self.insert(node, parent.children[distance])

    def query(self, log_content, threshold):
        """
        queryandgivensetlogcontentmostcloseofcluster ID
        """
        best_match = None
        best_distance = float("inf")

        def search(node, distance):
            nonlocal best_match, best_distance
            if distance <= threshold and distance < best_distance:
                best_match = node.log_id
                best_distance = distance

            for dist, child in node.children.items():
                if abs(dist - distance) <= threshold:
                    search(
                        child, self.distance_function(log_content, child.log_content)
                    )

        if self.root:
            search(
                self.root, self.distance_function(log_content, self.root.log_content)
            )

        return best_match

    def get_next_cluster_id(self):
        """
        getbottomoneitemscluster ID
        """
        cluster_id = self.next_cluster_id
        self.next_cluster_id += 1
        return cluster_id

    def find_node_by_cluster_id(self, cluster_id):
        """
        Recursively find BKTreeNode with specified cluster_id
        """

        def search_node(node):
            if node.log_id == cluster_id:
                return node
            for child in node.children.values():
                result = search_node(child)
                if result:
                    return result
            return None

        if self.root:
            return search_node(self.root)
        return None


def build_log_bktree(logs, distance_matrix, threshold=1.0):
    """
    structurebuildlog BKTree
    :param logs: loglist，eachitemslogisoneitemsstatesequence
    :param distance_matrix: statebetweenofDistance matrix
    :param threshold: distancethresholdvalue
    :return: log BKTree and logs to cluster ID mapping of
    """
    log_bktree = LogBKTree(lambda seq1, seq2: dtw_distance(seq1, seq2, distance_matrix))
    log_to_cluster_id = {}

    for log_id, log_content in enumerate(logs):
        # query BKTree，check if matching cluster exists
        matched_cluster_id = log_bktree.query(log_content, threshold)

        if matched_cluster_id is None:
            # if no matching cluster，create new cluster
            new_cluster_id = log_bktree.get_next_cluster_id()
            new_node = LogBKTreeNode(new_cluster_id, log_content)
            log_bktree.insert(new_node)
            log_to_cluster_id[log_id] = new_cluster_id
        else:
            # if matching cluster exists，map logassign tocluster
            log_to_cluster_id[log_id] = matched_cluster_id

    return log_bktree, log_to_cluster_id


def calculate_cluster_centroid(cluster_logs, state_distance_matrix):
    """
    calculateclustercenter（centroid）：clustercenterisclusterinallsequenceofaverage DTW distanceofrepresentativesequence
    :param cluster_logs: clusterinoflogsequencelist，each element is (sequence, fitness value, dataset identifier)
    :param state_distance_matrix: statebetweenofDistance matrix
    :return: clustercentersequence
    """
    if len(cluster_logs) == 1:
        return cluster_logs[0][0]  # if cluster has only one sequence，directlyreturnthesequenceasasclustercenter

    # extractallsequence
    sequences = [log[0] for log in cluster_logs]

    # calculateclusterinallsequenceofaverage DTW distance
    centroid_dtw_distances = np.zeros(len(sequences))
    for i in range(len(sequences)):
        total_dtw = 0
        for j in range(len(sequences)):
            if i != j:
                total_dtw += dtw_distance(
                    sequences[i], sequences[j], state_distance_matrix
                )
        centroid_dtw_distances[i] = total_dtw / (len(sequences) - 1)

    # select DTW distancemostsmallofsequenceasasclustercenter
    centroid_index = np.argmin(centroid_dtw_distances)
    return sequences[centroid_index]


def calculate_silhouette_score(
    cluster_representatives, state_distance_matrix, sample_size=100
):
    """
    calculateclusterofsilhouette coefficient，use distance between centroids as inter-cluster distance
    :param cluster_representatives: a dictionary，key iscluster ID，value is log list in cluster
    :param state_distance_matrix: statebetweenofDistance matrix
    :param sample_size: sample size for intra-cluster distance in each cluster
    :return: clusterofsilhouette coefficient
    """
    # calculateeachitemsclusterofclustercenter
    cluster_centroids = {}
    total_clusters = len(cluster_representatives)
    progress_threshold = max(1, total_clusters // 10)  # eachatmanage 10% ofclusteroutputonetimesenterdegree
    # initializetotal start time
    total_start_time = time.time()
    last_output_time = total_start_time

    def output_progress(index, total, last_output_time, total_start_time, message=""):
        current_time = time.time()
        time_elapsed = current_time - last_output_time
        total_elapsed_time = current_time - total_start_time  # calculatetotal time
        print(
            f"{message} {index + 1} out of {total} clusters ({(index + 1) / total * 100:.1f}%) "
            f"(Time elapsed: {time_elapsed:.2f} seconds, Total time: {total_elapsed_time:.2f} seconds)"
        )
        return current_time

    for index, (cluster_id, logs) in enumerate(cluster_representatives.items()):
        # calculateclustercenter
        cluster_centroids[cluster_id] = calculate_cluster_centroid(
            logs, state_distance_matrix
        )

        # output progress
        if (index + 1) % progress_threshold == 0 or index == total_clusters - 1:
            last_output_time = output_progress(
                index,
                total_clusters,
                last_output_time,
                total_start_time,
                message="Calculated centroid for",
            )

    # initializesilhouette coefficientlist
    silhouette_scores = []

    # traverse each cluster
    print("Calculating silhouette scores for clusters...")
    for index, (cluster_id, logs) in enumerate(cluster_representatives.items()):
        # if log count less than sample size，then use all logs directly
        sample_logs = random.sample(logs, min(sample_size, len(logs)))

        # extractallsequence
        sequences = [log[0] for log in sample_logs]

        # calculateclusterindistance
        centroid = cluster_centroids[cluster_id]
        a = np.mean(
            [dtw_distance(seq, centroid, state_distance_matrix) for seq in sequences]
        )

        # calculateclusterbetweendistance
        other_clusters = [
            c_id for c_id in cluster_representatives.keys() if c_id != cluster_id
        ]
        b = min(
            [
                dtw_distance(
                    cluster_centroids[cluster_id],
                    cluster_centroids[other_id],
                    state_distance_matrix,
                )
                for other_id in other_clusters
            ]
        )

        # calculatesilhouette coefficient
        silhouette = (b - a) / max(a, b) if max(a, b) > 0 else 0
        silhouette_scores.append(silhouette)

        progress_threshold = max(
            1, total_clusters // 100
        )  # eachatmanage 10% ofclusteroutputonetimesenterdegree
        # output progress
        if (index + 1) % progress_threshold == 0 or index == total_clusters - 1:
            last_output_time = output_progress(
                index,
                total_clusters,
                last_output_time,
                total_start_time,
                message="Processed",
            )

    # returnaveragesilhouette coefficient
    return round(np.mean(silhouette_scores), 3)


def cluster_logs_with_bktree(
    all_logs: List,
    all_results: List,
    all_data_set_ids: List,
    state_distance_matrix,
    threshold: float,
) -> Tuple[float, List[Tuple], Dict[int, List[Tuple]]]:
    log_bktree = LogBKTree(
        lambda seq1, seq2: dtw_distance(seq1, seq2, state_distance_matrix)
    )
    log_to_cluster_id = {}
    total_logs = len(all_logs)
    progress_threshold = max(1, total_logs // 10)  # ensureat leastoutputonetimes，and interval is10%
    last_output_time = time.time()
    total_start_time = last_output_time
    total_elapsed_times = []  # useatstoreeachtimesenterdegreeoutputwhenoftotal time

    for log_id, log in enumerate(all_logs):
        operation_start_time = time.time()  # record start time of current operation
        matched_cluster_id = log_bktree.query(log, threshold=threshold)
        if matched_cluster_id is None:
            new_cluster_id = log_bktree.get_next_cluster_id()
            new_node = LogBKTreeNode(new_cluster_id, log)
            log_bktree.insert(new_node)
            log_to_cluster_id[log_id] = new_cluster_id
        else:
            log_to_cluster_id[log_id] = matched_cluster_id
        operation_end_time = time.time()  # record end time of current operation

        # Check if progress output is needed
        if (log_id + 1) % progress_threshold == 0 or log_id == total_logs - 1:
            current_time = time.time()
            total_elapsed_time = current_time - total_start_time
            time_since_last_output = current_time - last_output_time
            total_elapsed_times.append(total_elapsed_time)
            print(
                f"Processed {log_id + 1} out of {total_logs} logs ({(log_id + 1) / total_logs * 100:.1f}%), "
                f"Time since last output: {time_since_last_output:.2f} seconds, Total elapsed time: {total_elapsed_time:.2f} seconds"
            )
            last_output_time = current_time

    # map log、result、cluster ID and dataset ID integrate
    clustered_data = [
        (log, result, log_to_cluster_id[log_id], data_set_id)
        for log_id, (log, result, data_set_id) in enumerate(
            zip(all_logs, all_results, all_data_set_ids)
        )
    ]

    # selecteachitemsclustermiddlefitness valuemostheightoflogasasrepresentative
    cluster_representatives = defaultdict(list)
    for log, result, cluster_id, data_set_id in clustered_data:
        cluster_representatives[cluster_id].append((log, result, data_set_id))

    print(
        f"Number of clusters with threshold {threshold}: {len(cluster_representatives)}"
    )
    silhouette_avg = calculate_silhouette_score(
        cluster_representatives, state_distance_matrix
    )
    print(f"Silhouette score with threshold {threshold}: {silhouette_avg}")

    return silhouette_avg, clustered_data, cluster_representatives


def cluster_logs_with_bktree_no_silhouette(
    all_logs: List,
    all_results: List,
    all_data_set_ids: List,
    state_distance_matrix,
    threshold: float,
) -> Tuple[List[Tuple], Dict[int, List[Tuple]]]:
    log_bktree = LogBKTree(
        lambda seq1, seq2: dtw_distance(seq1, seq2, state_distance_matrix)
    )
    log_to_cluster_id = {}
    total_logs = len(all_logs)
    progress_threshold = max(1, total_logs // 10)  # ensureat leastoutputonetimes，and interval is10%
    last_output_time = time.time()
    total_start_time = last_output_time
    total_elapsed_times = []  # useatstoreeachtimesenterdegreeoutputwhenoftotal time

    for log_id, log in enumerate(all_logs):
        operation_start_time = time.time()  # record start time of current operation
        matched_cluster_id = log_bktree.query(log, threshold=threshold)
        if matched_cluster_id is None:
            new_cluster_id = log_bktree.get_next_cluster_id()
            new_node = LogBKTreeNode(new_cluster_id, log)
            log_bktree.insert(new_node)
            log_to_cluster_id[log_id] = new_cluster_id
        else:
            log_to_cluster_id[log_id] = matched_cluster_id
        operation_end_time = time.time()  # record end time of current operation

        # Check if progress output is needed
        if (log_id + 1) % progress_threshold == 0 or log_id == total_logs - 1:
            current_time = time.time()
            total_elapsed_time = current_time - total_start_time
            time_since_last_output = current_time - last_output_time
            total_elapsed_times.append(total_elapsed_time)
            print(
                f"Processed {log_id + 1} out of {total_logs} logs ({(log_id + 1) / total_logs * 100:.1f}%), "
                f"Time since last output: {time_since_last_output:.2f} seconds, Total elapsed time: {total_elapsed_time:.2f} seconds"
            )
            last_output_time = current_time

    # map log、result、cluster ID and dataset ID integrate
    clustered_data = [
        (log, result, log_to_cluster_id[log_id], data_set_id)
        for log_id, (log, result, data_set_id) in enumerate(
            zip(all_logs, all_results, all_data_set_ids)
        )
    ]

    # selecteachitemsclustermiddlefitness valuemostheightoflogasasrepresentative
    cluster_representatives = defaultdict(list)
    for log, result, cluster_id, data_set_id in clustered_data:
        cluster_representatives[cluster_id].append((log, result, data_set_id))

    return clustered_data, cluster_representatives


def add_state_to_all_states(sample_size=4000):
    panorama_primary_bk_tree = BKTree(
        custom_distance.multi_distance, distance_index=0
    )
    panorama_secondary_bktree = defaultdict(
        lambda: BKTree(custom_distance.multi_distance, distance_index=1)
    )
    panorama_state_dict = {}
    panorama_state_dict_reverse = {}
    panorama_state_value_dict_reverse = {}
    state_maps = {}
    all_data = []
    # all_results = []
    data_segmentation = defaultdict(
        lambda: {
            "log_start": None,
            "log_end": None,
            "result_start": None,
            "result_end": None,
        }
    )
    log_index = 0
    for data_name, data_value in data_manager.items():
        state_maps[data_name] = {}
        primary_bk_tree = load_bk_tree_from_file(data_value["primary_bktree_path"])
        secondary_bk_trees = {}
        cluster_count = get_max_cluster_id(primary_bk_tree)
        for cluster_id in range(1, cluster_count + 1):
            secondary_bktree_path = (
                f"{data_value['secondary_bktree_prefix']}_{cluster_id}.json"
            )
            secondary_bk_trees[cluster_id] = load_bk_tree_from_file(
                secondary_bktree_path
            )

        with open(data_value["state_node_path"], "r") as f:
            for line in f:
                if line.strip():
                    line_info = line.strip().split("\t")
                    state_str = line_info[0]
                    cluster_id = int(line_info[1])
                    primary_id = int(state_str.strip("()").split(",")[0])
                    secondary_id = int(state_str.strip("()").split(",")[1])
                    norm_state = (
                        secondary_bk_trees.get(primary_id)
                        .find_node_by_cluster_id(secondary_id)
                        .state
                    )
                    panorama_state_str, state_value = get_state_cluster(
                        panorama_primary_bk_tree, panorama_secondary_bktree, norm_state
                    )
                    # print(state_value)
                    if panorama_state_str not in panorama_state_dict:
                        idx = len(panorama_state_dict)
                        panorama_state_dict[panorama_state_str] = idx
                        panorama_state_dict_reverse[idx] = {
                            "cluster": panorama_state_str
                        }
                        if idx not in panorama_state_value_dict_reverse:
                            panorama_state_value_dict_reverse[idx] = []
                        panorama_state_value_dict_reverse[idx].append(state_value)
                    panorama_state_id = panorama_state_dict[panorama_state_str]
                    state_maps[data_name][cluster_id] = panorama_state_id
        with (
            open(data_value["node_log_path"], "r") as f_log,
            open(data_value["game_result_path"], "r") as f_result,
        ):
            logs = [
                [state_maps[data_name][int(num)] for num in line.strip().split()]
                for line in f_log
                if line.strip()
            ]
            # logs = [list(map(int, line.strip().split())) for line in f_log if line.strip()]
            # logs = [state_maps[data_name][num] for num in [list(map(int, line.strip().split())) for line in f_log if line.strip()]]
            if data_name == "HRL_IMCBS":
                results = [
                    float(line.strip().split("\t")[2])
                    + float(line.strip().split("\t")[3])
                    for line in f_result
                    if line.strip()
                ]
            else:
                results = [
                    float(line.strip().split("\t")[1])
                    for line in f_result
                    if line.strip()
                ]

        # ensurelogandresultofcountonecause
        assert len(logs) == len(results), f"logandresultcountnotonecause: {data_name}"

        # if sample count greater thansample_size，then sample at uniform intervalssample_sizesamples
        if len(logs) > sample_size:
            step = len(logs) // sample_size
            sampled_logs = logs[::step]
            sampled_results = results[::step]
            # truncate tosample_size
            sampled_logs = sampled_logs[:sample_size]
            sampled_results = sampled_results[:sample_size]
        else:
            sampled_logs = logs
            sampled_results = results

        # map logandresultintegrateas (log, result) tuple list
        combined_data = list(
            zip(sampled_logs, sampled_results, [data_name] * len(sampled_logs))
        )

        print(f"data_name: {data_name}, number of logs: {len(combined_data)}")

        all_data.extend(combined_data)
        # # get sampling percentage
        # sample_ratio = sampling_manager.get(f"{data_name}", 1000) / 1000 # default to 100%
        #
        # # sort by fitness descending
        # sorted_data = sorted(combined_data, key=lambda x: x[1], reverse=True)
        #
        # # extract first5%andafter5%data ofindex
        # total_size = len(sorted_data)
        # top_5_percent_idx = int(total_size * 0.05)
        # bottom_5_percent_idx = int(total_size * 0.95)  # after5%start position of

        # # get first5%andafter5%data of
        # top_data = sorted_data[:top_5_percent_idx]
        # bottom_data = sorted_data[bottom_5_percent_idx:]
        #
        # # for first5%andafter5%uniform sampling separately
        # def sample_uniformly(data, target_size):
        #     if len(data) <= target_size:
        #         return data  # return all when data insufficient
        #     step = len(data) / target_size
        #     sampled_indices = [int(i * step) for i in range(target_size)]
        #     return [data[i] for i in sampled_indices]
        #
        # # calculateeachpartial samplingcount
        # total_samples_needed = int(total_size * sample_ratio)
        # top_samples_needed = int(total_samples_needed * 0.5)  # half before and after
        # bottom_samples_needed = total_samples_needed - top_samples_needed
        #
        # # execute sampling
        # sampled_top = sample_uniformly(top_data, top_samples_needed)
        # sampled_bottom = sample_uniformly(bottom_data, bottom_samples_needed)
        #
        # # merge sampling results
        # sampled_data = sampled_top + sampled_bottom
        #
        # # willsamplingafterdata ofaddto globallistmiddle
        # all_data.extend(sampled_data)
        #
        # # update data partition info
        # data_segmentation[data_name]['start'] = log_index
        # data_segmentation[data_name]['end'] = log_index + len(sampled_data)
        # log_index += len(sampled_data)

    # separate logs and results
    all_logs, all_results, all_data_set_ids = zip(*all_data)  # parsecontaintuple list

    state_distance_matrix = calculate_and_save_distance_matrix(
        panorama_state_dict_reverse,
        custom_distance,
        panorama_secondary_bktree,
        path_manager["distance_matrix_folder"],
    )

    # 2025-09-30
    # 1. first generate global 6 color table（only once）
    GLOBAL_COLOR_MAP = {"HRL_IMCBS": ALGO_COLOR["HRL_IMCBS"]}
    for i, algo in enumerate(ALL_ALGORITHMS):
        if algo not in GLOBAL_COLOR_MAP:
            GLOBAL_COLOR_MAP[algo] = COLOR_LIST[i - 1]

    # 2. collect information & plotting
    state_info = build_state_algorithm_info(
        all_logs,
        all_results,
        all_data_set_ids,
        panorama_state_dict_reverse,
        state_distance_matrix,
    )
    suffix = "_".join(
        [f"{key.lower()}_{value}" for key, value in scenario_manager.items()]
    )
    # target directory and file
    out_dir = f"{suffix}_{global_sample_size}"
    os.makedirs(out_dir, exist_ok=True)  # auto createbuildnested directories

    # drawdensityband
    # plot_fitness_frequency_with_kde_band(state_info, panorama_state_value_dict_reverse, save_path=os.path.join(out_dir, f'{scenario_manager['map']}_state_kde_band.pdf'))
    # plot_fitness_frequency_single_no_legend(state_info, panorama_state_value_dict_reverse, save_path=os.path.join(out_dir, f'{scenario_manager['map']}_state_kde_band_tight.pdf'))
    # plot_fitness_frequency_single_no_legend_sce_3_3m(state_info, panorama_state_value_dict_reverse, save_path=os.path.join(out_dir, f'{scenario_manager['map']}_state_kde_band_tight.pdf'))
    # plot_fitness_frequency_skew_band(state_info, panorama_state_value_dict_reverse, save_path=os.path.join(out_dir, f'state_skew_band_{global_sample_size}.pdf'), hdi_ratio=1)
    # plot_fitness_frequency_3d_kde_band(state_info, panorama_state_value_dict_reverse, bins=25, save_path=os.path.join(out_dir, f'{scenario_manager['map']}_3d_kde_band.pdf'), hdi_ratio=1)

    # plot_unique_states(state_info, state_distance_matrix, panorama_state_dict_reverse, save_path=os.path.join(out_dir, f'state_unique_{global_sample_size}'))
    # plot_unique_states_with_highlight(state_info, state_distance_matrix, panorama_state_dict_reverse, save_path=os.path.join(out_dir, f'state_unique_and_best_{global_sample_size}'))
    # plot_algorithm_best_states(state_info, state_distance_matrix, panorama_state_dict_reverse, save_path=os.path.join(out_dir, f'state_best_{global_sample_size}'))

    # withdrawstatevalueterrain
    plot_FL(
        state_info,
        state_distance_matrix,
        panorama_state_dict_reverse,
        panorama_state_value_dict_reverse,
        save_path=os.path.join(out_dir, f"{scenario_manager['map']}_state_landscape"),
    )
    # plot_unique_states_with_FL(state_info, state_distance_matrix, panorama_state_dict_reverse, panorama_state_value_dict_reverse, save_path=os.path.join(out_dir, f'state_unique_with_landscape_{global_sample_size}'))
    # plot_unique_states_with_highlight_with_FL(state_info, state_distance_matrix, panorama_state_dict_reverse, panorama_state_value_dict_reverse, save_path=os.path.join(out_dir, f'{scenario_manager['map']}_state_unique_and_best_with_landscape'))
    # plot_unique_states_with_highlight_with_FL_no_legend(state_info, state_distance_matrix, panorama_state_dict_reverse, panorama_state_value_dict_reverse, save_path=os.path.join(out_dir, f'{scenario_manager['map']}_state_unique_and_best_with_landscape_no_legend'))
    # plot_algorithm_best_states_with_FL(state_info, state_distance_matrix, panorama_state_dict_reverse, panorama_state_value_dict_reverse, save_path=os.path.join(out_dir, f'state_best_with_landscape_{global_sample_size}'))

    # ==============================log==============================

    # define different threshold value
    thresholds = [1.0, 1.5, 2.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0]
    # thresholds = [1.5]    # sce-1 1.5
    # k = len(thresholds)  # number of experiments
    #
    # # cycle k times，eachtimesuseusedifferentof threshold conduct experiment
    # for threshold in thresholds:
    #     print(f"Starting experiment with threshold: {threshold}")
    #     silhouette_avg, clustered_data, cluster_representatives = cluster_logs_with_bktree(all_logs, all_results, all_data_set_ids, state_distance_matrix, threshold)
    #     print(f"Experiment with threshold {threshold} completed. Silhouette score: {silhouette_avg}")

    for threshold in thresholds:
        # silhouette_avg, clustered_data, cluster_representatives = cluster_logs_with_bktree(all_logs, all_results, all_data_set_ids, state_distance_matrix, threshold)
        # print(f"Experiment with threshold {threshold} completed. Silhouette score: {silhouette_avg}")

        clustered_data, cluster_representatives = (
            cluster_logs_with_bktree_no_silhouette(
                all_logs,
                all_results,
                all_data_set_ids,
                state_distance_matrix,
                threshold,
            )
        )

        # extract representative log and result
        clustered_data = list(
            map(
                lambda value: value[0] if value else None,
                cluster_representatives.values(),
            )
        )

        clustered_logs, clustered_results, clustered_data_set_ids = zip(*clustered_data)

        cluster_log_distance_matrix = calculate_and_save_dtw_distance_matrix(
            clustered_logs,
            state_distance_matrix,
            path_manager["distance_matrix_folder"],
            threshold=threshold,
        )

        plot_fitness_landscape_3d(
            cluster_log_distance_matrix,
            clustered_results,
            clustered_data_set_ids,
            top_n=None,
        )

    # plot_all_fitness_landscapes(thresholds, all_logs, all_results, all_data_set_ids, state_distance_matrix, path_manager)

    log_distance_matrix = calculate_and_save_dtw_distance_matrix(
        all_logs, state_distance_matrix, path_manager["distance_matrix_folder"]
    )

    # plot_fitness_landscape_3d(log_distance_matrix, all_results, all_data_set_ids, top_n=None)


all_state = {}
# with open(data_manager["HRL_IMCBS"]['state_node_path'], 'r') as f:
#     for line in f:
#         if line.strip():
#             line_info = line.strip().split('\t')
#             state_str = line_info[0]
#             state_id = line_info[1]

add_state_to_all_states(sample_size=global_sample_size)
