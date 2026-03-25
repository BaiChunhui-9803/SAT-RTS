import json, math, os, re, time

import warnings
from matplotlib.patches import Rectangle

from config import (
    get_data_paths,
    get_output_dir,
    OutputPaths,
    DEFAULT_MAP_ID,
    DEFAULT_DATA_ID,
)

warnings.filterwarnings(
    "ignore", category=DeprecationWarning, module=r"plotly\.io\._kaleido"
)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="plotly")

from sklearn.manifold import MDS
import plotly.graph_objects as go
import plotly.io as pio
from src.distance.base import CustomDistance
from scipy.interpolate import griddata
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering, Birch
from sklearn.mixture import GaussianMixture
from matplotlib.colors import LinearSegmentedColormap
from src.analysis.pattern import *
from collections import defaultdict, Counter, OrderedDict
from mpl_toolkits.axes_grid1 import make_axes_locatable

custom_distance = CustomDistance(threshold=0.5)

# Dataset configuration - switch datasets by modifying here
# Available datasets: MarineMicro_MvsM_4 (data_ids: 1, 6), MarineMicro_MvsM_4_dist (data_id: 1), MarineMicro_MvsM_8 (data_id: 1)
map_id = DEFAULT_MAP_ID  # Or specify directly e.g., "MarineMicro_MvsM_4"
data_id = DEFAULT_DATA_ID  # Or specify directly e.g., "6"
# map_id = "MarineMicro_MvsM_4_dist"
# data_id = "1"

# Get data paths from configuration module
paths = get_data_paths(map_id, data_id)
distance_matrix_folder = paths["distance_matrix_folder"]
primary_bktree_path = paths["primary_bktree_path"]
secondary_bktree_prefix = paths["secondary_bktree_prefix"]
state_node_path = paths["state_node_path"]
node_log_path = paths["node_log_path"]
game_result_path = paths["game_result_path"]
action_log_path = paths["action_log_path"]
action_path = paths["action_path"]

# Custom color list
custom_colors = [
    "#bf0060",
    "#bfbf00",
    "#00bfbf",
    "#18BF00",
    "#8800bf",
    "#7b3c00",
    "#003e94",
    "#006630",
]


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
        Recursively find BKTreeNode with the specified cluster_id
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

def load_bk_tree_from_file(file_path):
    """
    Load BKTree data from file and restore as BKTree instance

    :param file_path: Path to the file
    :return: BKTree instance
    """

    def deserialize_node(node_data):
        """
        Recursively deserialize node
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
    Recursively find the maximum cluster_id in BKTree
    """
    if node.cluster_id > max_cluster_id[0]:
        max_cluster_id[0] = node.cluster_id

    for child in node.children.values():
        find_max_cluster_id(child, max_cluster_id)


def get_max_cluster_id(bk_tree):
    """
    Get the maximum cluster_id in BKTree
    """
    max_cluster_id = [0]  # Use list to store max value for modification during recursion
    if bk_tree.root:
        find_max_cluster_id(bk_tree.root, max_cluster_id)
    return max_cluster_id[0]


def read_state_node_file(file_path):
    """
    Read file and store as a dictionary
    :param file_path: Path to the file
    :return: Data in dictionary format
    """
    state_node_dict = {}
    with open(file_path, "r") as file:
        for line in file:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue  # Skip improperly formatted lines
            key = eval(parts[0])
            id = int(parts[1])
            score = float(parts[2])
            state_node_dict[key] = {"id": id, "score": score}

    reverse_dict = {}
    for key, value in state_node_dict.items():
        id = value["id"]
        score = value["score"]
        if id not in reverse_dict:
            reverse_dict[id] = {"cluster": key, "score": score}

    return state_node_dict, reverse_dict


def read_node_log_file(file_path):
    """
    Read file and save each line as a list, finally stored in a large list
    :param file_path: Path to the file
    :return: List containing all lines, where each line is also a list
    """
    result = []
    with open(file_path, "r") as file:
        for line in file:
            parts = line.strip().split()
            parts = [int(part) for part in parts]
            result.append(parts)
    return result


def read_game_result_file(file_path):
    """
    Read game_result.txt file and store results in a list

    Args:
        file_path: File path
    Returns:
        List containing all results, each row is also a list
    """
    result = []
    with open(file_path, "r") as file:
        for line in file:
            parts = line.strip().split("\t")
            if len(parts) < 4:
                continue  # Skip improperly formatted lines
            outcome = parts[0]
            steps = int(
                float(parts[1].strip("[]"))
            )  # Convert string '[180]' to list [180] then to int
            score = int(parts[2])
            penalty = int(parts[3])
            result.append([outcome, steps, score, penalty])
    return result


def calculate_distance_matrix(reverse_dict, custom_distance, secondary_bk_trees):
    """
    Calculate distance matrix
    :param reverse_dict: Reverse dictionary
    :param custom_distance: Custom distance calculation function
    :param secondary_bk_trees: Dictionary of secondary BKtrees
    :return: Distance matrix
    """
    # Get the number of all clusters
    num_clusters = len(reverse_dict)

    # Initialize distance matrix
    distance_matrix = np.zeros((num_clusters, num_clusters))

    # Get the status of all clusters
    clusters = list(reverse_dict.values())

    # Initialize time of last output
    last_output_time = time.time()

    # Initialize progress threshold
    progress_threshold = 0.01  # 1%

    # Calculate distance between every pair of clusters
    for i in range(num_clusters):
        for j in range(i + 1, num_clusters):  # Start from i+1 to avoid redundant calculation
            state1 = clusters[i]["cluster"]
            state2 = clusters[j]["cluster"]

            # Get nodes for two states
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

            # Fill distance matrix
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
            progress_threshold += 0.01  # Update progress threshold

    # Diagonal distance is 0
    for i in range(num_clusters):
        distance_matrix[i, i] = 0

    return distance_matrix


def dtw_distance(seq1, seq2, distance_matrix):
    """
    Calculate DTW distance between two sequences
    :param seq1: First sequence
    :param seq2: Second sequence
    :param distance_matrix: Distance matrix
    :return: DTW distance
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
    Calculate DTW distance matrix between all sequences
    :param state_log: List containing all sequences
    :param distance_matrix: Distance matrix
    :return: DTW distance matrix
    """
    num_sequences = len(state_log)
    dtw_distance_matrix = np.zeros((num_sequences, num_sequences))

    # Initialize time of last output
    last_output_time = time.time()

    # Initialize progress threshold
    progress_threshold = 0.01  # 1%

    # Calculate DTW distance between every pair of sequences
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
            progress_threshold += 0.01  # Update progress threshold

    return dtw_distance_matrix


def save_distance_matrix(matrix, file_path):
    """
    Save distance matrix to file
    :param matrix: Distance matrix
    :param file_path: Path to the file
    """
    np.save(file_path, matrix)


def load_distance_matrix(file_path):
    """
    Load distance matrix from file
    :param file_path: Path to the file
    :return: Distance matrix
    """
    return np.load(file_path)


def calculate_and_save_distance_matrix(
    reverse_dict, custom_distance, secondary_bk_trees, distance_matrix_folder
):
    """
    Calculate distance matrix and save to file
    :param reverse_dict: Reverse dictionary
    :param custom_distance: Custom distance calculation function
    :param secondary_bk_trees: Dictionary of secondary BKtrees
    :param distance_matrix_folder: Path to folder for saving distance matrix
    :return: Distance matrix
    """
    # Ensure folder exists
    if not os.path.exists(distance_matrix_folder):
        os.makedirs(distance_matrix_folder)
        print(f"Created directory: {distance_matrix_folder}")

    # Define matrix file path
    state_distance_matrix_path = os.path.join(
        distance_matrix_folder, "state_distance_matrix.npy"
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
    state_log, distance_matrix, dtw_distance_matrix_folder
):
    """
    Calculate DTW distance matrix and save to file
    :param state_log: List containing all sequences
    :param distance_matrix: Distance matrix
    :param dtw_distance_matrix_folder: Path to folder for saving DTW distance matrix
    :return: DTW distance matrix
    """
    # Ensure folder exists
    if not os.path.exists(dtw_distance_matrix_folder):
        os.makedirs(dtw_distance_matrix_folder)
        print(f"Created directory: {dtw_distance_matrix_folder}")

    # Define matrix file path
    log_distance_matrix_path = os.path.join(
        dtw_distance_matrix_folder, "log_distance_matrix.npy"
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


def plot_fitness_landscape(log_distance_matrix, log_object, game_results, top_n=None):
    """
    Draw fitness landscape map
    :param log_distance_matrix: Distance matrix between logs
    :param log_object: Fitness value for each log
    :param game_results: List of game results
    :param top_n: Select top_n logs to draw; if None, draw all data
    """
    # If top_n is None, draw all data
    if top_n is None:
        top_n = len(log_object)

    # Only take distance matrix and fitness values for the top_n logs
    log_distance_matrix = log_distance_matrix[:top_n, :top_n]
    log_object = log_object[:top_n]
    game_results = game_results[:top_n]

    # Use MDS to map logs to a 2D plane
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
    log_positions = mds.fit_transform(log_distance_matrix)

    # Prepare grid required for interpolation
    grid_x, grid_y = np.mgrid[
        log_positions[:, 0].min() : log_positions[:, 0].max() : 100j,
        log_positions[:, 1].min() : log_positions[:, 1].max() : 100j,
    ]

    # Use griddata for interpolation
    grid_z = griddata(
        points=log_positions,
        values=log_object,
        xi=(grid_x, grid_y),
        method="linear",  # Can choose 'linear', 'nearest', 'cubic'
    )

    # Draw the interpolated landscape map
    plt.figure(figsize=(10, 8))
    plt.contourf(grid_x, grid_y, grid_z, cmap="coolwarm", levels=100)
    plt.colorbar(label="Fitness Value")

    # Plot sample points
    # scatter = plt.scatter(log_positions[:, 0], log_positions[:, 1], c=log_object, cmap='coolwarm', s=100, edgecolor='k')

    # Add title and labels
    plt.title("Fitness Landscape")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")

    # Add legend
    plt.legend()
    plt.show()


def plot_fitness_landscape_3d(
    log_distance_matrix, log_object, game_results, top_n=None
):
    """
    Draw fitness landscape map in 3D
    :param log_distance_matrix: Distance matrix between logs
    :param log_object: Fitness value for each log
    :param game_results: List of game results
    :param top_n: Select top_n logs to draw; if None, draw all data
    """
    # If top_n is None, draw all data
    if top_n is None:
        top_n = len(log_object)

    # Only take distance matrix and fitness values for the top_n logs
    log_distance_matrix = log_distance_matrix[:top_n, :top_n]
    log_object = log_object[:top_n]
    game_results = game_results[:top_n]

    # Use MDS to map logs to a 2D plane
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
    log_positions = mds.fit_transform(log_distance_matrix)

    # Prepare grid required for interpolation
    grid_x, grid_y = np.mgrid[
        log_positions[:, 0].min() : log_positions[:, 0].max() : 100j,
        log_positions[:, 1].min() : log_positions[:, 1].max() : 100j,
    ]

    # Use griddata for interpolation
    grid_z = griddata(
        points=log_positions,
        values=log_object,
        xi=(grid_x, grid_y),
        method="linear",  # Can choose 'linear', 'nearest', 'cubic'
    )

    # Calculate position of 0 in the color bar
    min_val = np.min(log_object)
    max_val = np.max(log_object)
    range_val = max_val - min_val

    if min_val >= 0 or max_val <= 0:
        # If all positive or all negative, set 0 in the middle of color bar
        zero_position = 0.5
    else:
        # Calculate ratio of 0 between positive and negative values
        zero_position = (0 - min_val) / range_val

    # Custom color scale fixing 0 value at specified position with white color
    custom_colorscale = [
        [0, "#5c7ee6"],  # Min value corresponds to blue
        [zero_position, "#ebebeb"],  # 0 value corresponds to white
        [1, "#b62d0a"],  # Max value corresponds to red
    ]

    # Draw interpolated 3D landscape map
    fig = go.Figure()

    # Add 3D surface
    fig.add_trace(
        go.Surface(
            x=grid_x,
            y=grid_y,
            z=grid_z,
            colorscale=custom_colorscale,
            showscale=True,
            showlegend=False,  # Do not show legend
            colorbar=dict(title="Fitness Value"),
        )
    )

    # Add sample points
    sample_points = go.Scatter3d(
        x=log_positions[:, 0],
        y=log_positions[:, 1],
        z=log_object,
        mode="markers",
        marker=dict(
            size=5,
            color=log_object,
            colorscale=custom_colorscale,
            line=dict(  # Add border settings
                color="black", width=1
            ),
        ),
        name="Sample Points",
        visible=True,  # Visible by default
    )
    fig.add_trace(sample_points)

    # Add red points for global optimal solutions
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
                visible=False,  # Hidden by default
                showlegend=False,  # Do not show legend
            )
        )
    fig.add_traces(global_optimum_points)

    # Add buttons to switch display modes
    buttons = [
        dict(
            label="Sample Points View / Global Optimum View",
            method="update",
            args=[
                {"visible": [True, False] + [True] * len(global_optimum_points)}
            ],  # Show global optimum
            args2=[
                {"visible": [True, True] + [False] * len(global_optimum_points)}
            ],  # Show sample points
        )
    ]

    # Calculate max and min for log_positions
    x_min, x_max = log_positions[:, 0].min(), log_positions[:, 0].max()
    y_min, y_max = log_positions[:, 1].min(), log_positions[:, 1].max()
    # Calculate max and min for log_object
    z_min, z_max = min(log_object), max(log_object)

    # Update layout, dynamically set ranges for x and y
    fig.update_layout(
        title="Fitness Landscape",
        scene=dict(
            xaxis_title="Dimension 1",
            yaxis_title="Dimension 2",
            zaxis_title="Fitness Value",
            aspectmode="manual",  # Set ratio manually
            aspectratio=dict(x=1, y=1, z=0.75),  # Adjust Z-axis ratio
            xaxis_range=[x_min - 5, x_max + 5],  # Dynamically set x-axis range
            yaxis_range=[y_min - 5, y_max + 5],  # Dynamically set y-axis range
            zaxis_range=[z_min - 3, z_max + 3],  # Dynamically set z-axis range
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
    )

    # Show figure
    fig.show()


def get_optimal_solution_distances(
    state_distance_matrix,
    log_distance_matrix,
    log_object,
    game_results,
    state_log,
    top_n=None,
):
    """
    Get distances between all states contained in optimal solutions, and distances between optimal solution sequences
    :param state_distance_matrix: Distance matrix between states
    :param log_distance_matrix: Distance matrix between log sequences
    :param log_object: Fitness value for each log
    :param game_results: List of game results
    :param top_n: Select top_n logs for calculation; if None, use all data
    :return: State distance matrix for optimal solutions, log sequence distance matrix for optimal solutions
    """
    # If top_n is None, use all data
    if top_n is None:
        top_n = len(log_object)

    # Only take fitness values and game results for top_n logs
    log_object = log_object[:top_n]
    game_results = game_results[:top_n]

    # Find indices of optimal solutions
    max_fitness = np.max(log_object)
    optimal_indices = np.where(log_object == max_fitness)[0]

    # Extract log sequences and results corresponding to optimal solutions
    optimal_logs = [state_log[i] for i in optimal_indices]
    optimal_logs_result = [game_results[i] for i in optimal_indices]

    # Extract all states in the optimal solution sequences
    optimal_states = []
    for log in optimal_logs:
        optimal_states.extend(log)

    optimal_states = sorted(list(set(optimal_states)), reverse=False)

    # Convert optimal_states to NumPy array
    optimal_states_np = np.array(optimal_states)

    # Extract distance matrix between optimal solution states
    optimal_state_distance_matrix = state_distance_matrix[
        np.ix_(optimal_states_np, optimal_states_np)
    ]

    # Extract distance matrix between optimal solution sequences
    optimal_log_distance_matrix = log_distance_matrix[
        np.ix_(optimal_indices, optimal_indices)
    ]

    return (
        optimal_state_distance_matrix,
        optimal_log_distance_matrix,
        optimal_indices,
        optimal_logs,
        optimal_states,
    )


def get_top_k_solution_distances(
    state_distance_matrix,
    log_distance_matrix,
    log_object,
    game_results,
    state_log,
    top_n=None,
    topk=None,
):
    """
    Get distances between states contained in logs with top topk percentage fitness, and distances between these log sequences
    :param state_distance_matrix: Distance matrix between states
    :param log_distance_matrix: Distance matrix between log sequences
    :param log_object: Fitness value for each log
    :param game_results: List of game results
    :param state_log: State sequence corresponding to each log
    :param top_n: Select top_n logs for calculation; if None, use all data
    :param topk: Select logs with top topk percentage fitness; topk is a value between 0 and 1
    :return: State distance matrix for top topk logs, distance matrix for these log sequences, and positions of these logs
    """
    # If top_n is None, use all data
    if top_n is None:
        top_n = len(log_object)

    from config import get_cache_path

    log_positions_file = get_cache_path(f"log_positions_{map_id}_{data_id}.npy")
    # Check if log_positions file exists
    if os.path.exists(log_positions_file):
        print(f"Loading saved log positions file: {log_positions_file}")
        log_positions = np.load(log_positions_file)
    else:
        print("Calculating log positions and saving to file...")
        # Use MDS to map logs to a 2D plane
        mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
        log_positions = mds.fit_transform(log_distance_matrix)
        # Save log_positions to file
        np.save(log_positions_file, log_positions)

    # Only take fitness values and game results for top_n logs
    log_object = log_object[:top_n]
    game_results = game_results[:top_n]
    state_log = state_log[:top_n]
    log_positions = log_positions[:top_n]  # Limit log_positions range

    # Sort logs based on fitness values
    sorted_indices = np.argsort(log_object)[::-1]  # Sort descending

    # If topk is not None, only take the top topk percentage of logs
    if topk is not None:
        num_topk = int(top_n * topk)  # Calculate number of logs to select
        sorted_indices = sorted_indices[:num_topk]

    # Extract indices, log sequences, and results for top topk percentage logs
    topk_indices = sorted_indices
    topk_logs = [state_log[i] for i in topk_indices]
    topk_logs_result = [game_results[i][2] + game_results[i][3] for i in topk_indices]

    # Extract all states in these log sequences
    topk_states = []
    for log in topk_logs:
        topk_states.extend(log)

    # Remove duplicates and sort
    topk_states = sorted(list(set(topk_states)), reverse=False)

    # Convert topk_states to NumPy array
    topk_states_np = np.array(topk_states)

    # Extract distance matrix between these states
    topk_state_distance_matrix = state_distance_matrix[
        np.ix_(topk_states_np, topk_states_np)
    ]

    # Extract distance matrix between these log sequences
    topk_log_distance_matrix = log_distance_matrix[np.ix_(topk_indices, topk_indices)]

    # Extract positions of these logs
    topk_log_positions = log_positions[topk_indices]

    return (
        topk_state_distance_matrix,
        topk_log_distance_matrix,
        topk_indices,
        topk_logs,
        topk_states,
        topk_log_positions,
        topk_logs_result,
    )


def plot_state_transition_graph(
    optimal_state_distance_matrix,
    optimal_log_distance_matrix,
    optimal_logs,
    optimal_states,
    kmeans_labels,
    k=3,
    width=350,
    height=600,
):
    """
    Draw state transition graph between states using Plotly and save as image
    :param optimal_state_distance_matrix: State distance matrix for optimal solutions
    :param optimal_log_distance_matrix: Log sequence distance matrix
    :param optimal_logs: Log sequences corresponding to optimal solutions
    :param optimal_states: List of states in optimal solutions
    :param kmeans_labels: K-Means clustering results
    :param k: Number of clusters
    """
    # Get number of states
    num_states = len(optimal_states)

    # Use MDS to distribute states on a 2D plane following distance relationships
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
    state_positions = mds.fit_transform(optimal_state_distance_matrix)

    # Create state nodes
    state_nodes = go.Scatter(
        x=state_positions[:, 0],
        y=state_positions[:, 1],
        mode="markers",
        text=optimal_states,  # Use state ID as text
        textposition="top center",
        marker=dict(size=10, color="white", line=dict(width=2, color="black")),
    )

    # Create directed edges for state transitions (Main graph)
    edge_x = []
    edge_y = []
    drawn_edges = set()  # Track already drawn edges

    for log in optimal_logs:
        for i in range(len(log) - 1):
            start_state = log[i]
            end_state = log[i + 1]
            start_index = optimal_states.index(start_state)
            end_index = optimal_states.index(end_state)
            edge = (start_index, end_index)

            if edge not in drawn_edges:
                edge_x.append(state_positions[start_index, 0])
                edge_x.append(state_positions[end_index, 0])
                edge_x.append(None)
                edge_y.append(state_positions[start_index, 1])
                edge_y.append(state_positions[end_index, 1])
                edge_y.append(None)
                drawn_edges.add(edge)

    state_edges = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=2, color="black"),
        hoverinfo="none",
    )

    # Create main graph
    fig_main = go.Figure(data=[state_edges, state_nodes])
    fig_main.update_layout(
        showlegend=False,
        hovermode="closest",
        margin=dict(b=0, l=0, r=0, t=0),
        xaxis=dict(
            showticklabels=False,  # Do not show x-axis ticks
            showgrid=False,  # Do not show x-axis grid
            zeroline=False,  # Do not show x-axis zero line
            visible=False,  # Do not show x-axis
        ),
        yaxis=dict(
            showticklabels=False,  # Do not show y-axis ticks
            showgrid=False,  # Do not show y-axis grid
            zeroline=False,  # Do not show y-axis zero line
            visible=False,  # Do not show y-axis
        ),
        plot_bgcolor="white",  # Set plot background to white
        paper_bgcolor="white",  # Set paper background to white
    )

    # Show main graph
    # fig_main.show()
    pio.write_image(
        fig_main,
        f"main_state_transition_graph.pdf",
        width=width,
        height=height,
        scale=4,
    )

    # Draw one graph for each cluster
    for cluster_id in range(k):
        cluster_logs = [
            optimal_logs[i]
            for i in range(len(optimal_logs))
            if kmeans_labels[i] == cluster_id
        ]

        # Create directed edges for state transitions (Cluster graph)
        edge_x_cluster = []
        edge_y_cluster = []
        edge_x_other = []
        edge_y_other = []

        # Track already drawn edges
        drawn_edges = set()

        for log in cluster_logs:
            for i in range(len(log) - 1):
                start_state = log[i]
                end_state = log[i + 1]
                start_index = optimal_states.index(start_state)
                end_index = optimal_states.index(end_state)
                edge = (start_index, end_index)

                if edge not in drawn_edges:
                    edge_x_cluster.append(state_positions[start_index, 0])
                    edge_x_cluster.append(state_positions[end_index, 0])
                    edge_x_cluster.append(None)
                    edge_y_cluster.append(state_positions[start_index, 1])
                    edge_y_cluster.append(state_positions[end_index, 1])
                    edge_y_cluster.append(None)
                    drawn_edges.add(edge)

        # Draw other edges not yet drawn
        for log in optimal_logs:
            for i in range(len(log) - 1):
                start_state = log[i]
                end_state = log[i + 1]
                start_index = optimal_states.index(start_state)
                end_index = optimal_states.index(end_state)
                edge = (start_index, end_index)

                if edge not in drawn_edges:
                    edge_x_other.append(state_positions[start_index, 0])
                    edge_x_other.append(state_positions[end_index, 0])
                    edge_x_other.append(None)
                    edge_y_other.append(state_positions[start_index, 1])
                    edge_y_other.append(state_positions[end_index, 1])
                    edge_y_other.append(None)
                    drawn_edges.add(edge)

        state_edges_cluster = go.Scatter(
            x=edge_x_cluster,
            y=edge_y_cluster,
            mode="lines",
            line=dict(width=2, color=f"{custom_colors[cluster_id]}"),
            hoverinfo="none",
        )

        state_edges_other = go.Scatter(
            x=edge_x_other,
            y=edge_y_other,
            mode="lines",
            line=dict(width=2, color="#d4d4d4"),
            hoverinfo="none",
        )

        # Create cluster graph
        fig_cluster = go.Figure(
            data=[state_edges_other, state_edges_cluster, state_nodes]
        )
        fig_cluster.update_layout(
            showlegend=False,
            hovermode="closest",
            margin=dict(b=0, l=0, r=0, t=0),
            xaxis=dict(
                showticklabels=False,  # Do not show x-axis ticks
                showgrid=False,  # Do not show x-axis grid
                zeroline=False,  # Do not show x-axis zero line
                visible=False,  # Do not show x-axis
            ),
            yaxis=dict(
                showticklabels=False,  # Do not show y-axis ticks
                showgrid=False,  # Do not show y-axis grid
                zeroline=False,  # Do not show y-axis zero line
                visible=False,  # Do not show y-axis
            ),
            plot_bgcolor="white",  # Set plot background to white
            paper_bgcolor="white",  # Set paper background to white
        )

        # Save cluster graph as image with specified width and height
        pio.write_image(
            fig_cluster,
            f"cluster_{cluster_id + 1}_state_transition_graph.pdf",
            width=width,
            height=height,
            scale=4,
        )


def get_fitness_landscape(
    log_distance_matrix,
    log_object,
    n=None,
    filename=None,
):
    """
    Reduce dimension to 2D using MDS based on distance matrix, interpolate fitness values, and return 2D fitness landscape data
    :param log_distance_matrix: Distance matrix between log sequences
    :param log_object: List containing fitness values
    :param n: Number of samples for testing (optional)
    :param filename: Filename for saving calculation results
    :return: grid_x, grid_y, grid_z
    """
    from config import get_cache_path

    if filename is None:
        filename = get_cache_path(f"fitness_landscape_data_{map_id}_{data_id}.npy")

    # If n is specified, only take the first n samples
    if n is not None:
        log_distance_matrix = log_distance_matrix[:n, :n]
        log_object = log_object[:n]

    # Check if a file already exists for saved results
    if os.path.exists(filename):
        print(f"Loading data from {filename}...")
        data = np.load(filename)
        grid_x, grid_y, grid_z = data
    else:
        print("Computing MDS and interpolation...")
        # Reduce dimension to 2D using MDS
        mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
        mds_coords = mds.fit_transform(log_distance_matrix)

        # Prepare interpolation data
        x = mds_coords[:, 0]
        y = mds_coords[:, 1]
        z = log_object  # Fitness values

        # Define interpolation grid
        grid_x, grid_y = np.mgrid[x.min() : x.max() : 1000j, y.min() : y.max() : 1000j]

        # Use interpolation to estimate fitness values on the grid
        grid_z = griddata((x, y), z, (grid_x, grid_y), method="linear")

        # Save results to file
        data = np.array([grid_x, grid_y, grid_z])
        np.save(filename, data)
        print(f"Data saved to {filename}.")

    return grid_x, grid_y, grid_z


def kmeans_clustering(part_log_distance_matrix, part_positions, part_results, k=3):
    # Use K-Means clustering
    kmeans = KMeans(n_clusters=k, random_state=42)  # Specify number of clusters as 3
    kmeans_labels = kmeans.fit_predict(
        part_positions
    )  # Clustering using top_k_positions
    return kmeans_labels


def visualize_part_distances(
        part_log_distance_matrix,
        part_positions,
        part_results,
        grid_x,
        grid_y,
        grid_z,
        zero_position=0.5,
        k=3,
        algorithm="spectral",
        save_fig=False,
        enable_heatmap=False,
):
    """
    Visualize the distance matrix between log sequences and the 2D distribution of log positions,
    and plot the fitness landscape.
    :param part_log_distance_matrix: Distance matrix between log sequences
    :param part_positions: 2D coordinates of log positions
    :param part_results: List containing numerical results, used for color assignment
    :param grid_x: x-coordinate grid for the fitness landscape
    :param grid_y: y-coordinate grid for the fitness landscape
    :param grid_z: Fitness value grid for the fitness landscape
    :param zero_position: Position of the 0 value in the colorbar (between 0 and 1)
    :param k: Number of clusters
    :param algorithm: Clustering algorithm name ('kmeans', 'spectral', 'agglomerative', 'gmm', 'birch')
    :param save_fig: Whether to save the images
    :param enable_heatmap: Whether to calculate and plot the heatmap
    :return: cluster_labels
    """
    # Assign colors based on part_results values (used for both heatmap and fitness landscape)
    custom_col_colors = ["#c9725b", "#b7310f"]
    unique_results = np.unique(part_results)  # Get all unique categories
    n_classes = len(unique_results)  # Actual number of categories
    color_palette = sns.color_palette("Reds", n_classes)  # Generate n_classes gradient colors
    lut = dict(zip(unique_results, color_palette))  # Category -> RGB triplet
    col_colors = np.array([lut[result] for result in part_results])

    # Initialize heatmap related variables
    linkage_matrix = None
    if enable_heatmap:
        # Convert symmetric distance matrix to condensed form (for scipy linkage function)
        condensed_distance_matrix = squareform(part_log_distance_matrix)
        # Calculate linkage matrix for hierarchical clustering
        linkage_matrix = linkage(condensed_distance_matrix, method="average")

    # Perform clustering based on the selected algorithm
    if algorithm == "kmeans":
        clustering = KMeans(n_clusters=k, random_state=42)
        cluster_labels = clustering.fit_predict(part_positions)
    elif algorithm == "spectral":
        clustering = SpectralClustering(
            n_clusters=k,
            random_state=42,
            affinity="nearest_neighbors",
            n_neighbors=10
        )
        cluster_labels = clustering.fit_predict(part_positions)
    elif algorithm == "agglomerative":
        clustering = AgglomerativeClustering(n_clusters=k, linkage="ward")
        cluster_labels = clustering.fit_predict(part_positions)
    elif algorithm == "gmm":
        # Gaussian Mixture Model - suitable for elliptical clusters
        gmm = GaussianMixture(n_components=k, random_state=42, covariance_type="full")
        cluster_labels = gmm.fit_predict(part_positions)
    elif algorithm == "birch":
        # BIRCH - suitable for large-scale datasets
        clustering = Birch(n_clusters=k, threshold=0.5)
        cluster_labels = clustering.fit_predict(part_positions)
    else:
        available_algorithms = ["kmeans", "spectral", "agglomerative", "gmm", "birch"]
        raise ValueError(
            f"Unsupported clustering algorithm: {algorithm}. Available algorithms: {', '.join(available_algorithms)}"
        )

    # Create output directories
    output_dir = get_output_dir(OutputPaths.FITNESS_STANDARD)
    if not os.path.exists(f"{output_dir}/heatmap"):
        os.makedirs(f"{output_dir}/heatmap")
    if not os.path.exists(f"{output_dir}/landscape"):
        os.makedirs(f"{output_dir}/landscape")

    if save_fig and enable_heatmap:
        # Assign colors to dendrogram samples (based on clustering results)
        unique_clusters = np.unique(cluster_labels)
        cluster_lut = dict(zip(unique_clusters, custom_colors))  # Use custom colors

        # Assign colors to heatmap rows based on clustering results
        row_colors = np.array([cluster_lut[label] for label in cluster_labels])

        # Plot heatmap with row_colors
        plt.figure(figsize=(9, 8))
        g_with_row_colors = sns.clustermap(
            part_log_distance_matrix,
            annot=False,
            cmap="Blues_r",
            row_linkage=linkage_matrix,
            col_linkage=linkage_matrix,
            dendrogram_ratio=(0.12, 0),
            col_colors=col_colors,
            row_colors=row_colors,  # Add row color bar
            cbar_pos=(0.03, 0.07, 0.02, 0.18),
            cbar_kws={"label": "Distance"},
            tree_kws={"colors": "black", "linewidths": 1.2},
            figsize=(9, 8),
        )
        g_with_row_colors.ax_heatmap.set_xlabel("")  # Hide X-axis title
        g_with_row_colors.ax_heatmap.set_ylabel("")  # Hide Y-axis title
        g_with_row_colors.ax_heatmap.tick_params(
            axis="both", which="both", length=0, labelbottom=False, labelright=False
        )
        g_with_row_colors.cax.yaxis.set_label_position("left")  # Place label on the left
        g_with_row_colors.ax_heatmap.text(
            -0.075, 1.02, "Fitness Value", fontsize=10, color="black", ha="center", va="center",
            transform=g_with_row_colors.ax_heatmap.transAxes,
        )

        # Save heatmap with row_colors
        heatmap_with_colors_path = os.path.join(
            output_dir, f"heatmap/heatmap_with_row_colors_{algorithm}_k{k}.png"
        )
        # SAT-RTS: plot -> heatmap/heatmap_with_row_colors_{algorithm}_k{k}.png
        plt.savefig(heatmap_with_colors_path, dpi=150, bbox_inches="tight")
        plt.close()

        # Plot heatmap without row_colors
        plt.figure(figsize=(16, 8))
        # (Assuming the logic continues to save heatmap_without_colors_path as seen in previous context)
        heatmap_without_colors_path = os.path.join(
            output_dir, f"heatmap/heatmap_without_row_colors_{algorithm}_k{k}.png"
        )
        # SAT-RTS: plot -> heatmap/heatmap_without_row_colors_{algorithm}_k{k}.png
        plt.savefig(heatmap_without_colors_path, dpi=150, bbox_inches="tight")
        plt.close()

    # Plot Fitness Landscape
    plt.figure(figsize=(10, 8))
    # Custom colormap fixing 0 value at specified position with white color
    custom_cmap = LinearSegmentedColormap.from_list(
        "custom_coolwarm",
        [
            (0, "#5c7ee6"),  # Min value to Blue
            (zero_position, "#ffffff"),  # 0 value to White
            (1, "#b62d0a"),  # Max value to Red
        ],
    )

    # Plot filled contour map
    contour = plt.contourf(grid_x, grid_y, grid_z, cmap=custom_cmap, levels=100, zorder=1)
    # Add colorbar for the landscape
    cbar = plt.colorbar(contour)
    cbar.set_label("Fitness Value")

    # Plot scatter points for different clusters on the landscape
    for cluster_id in range(k):
        # Extract coordinates for current cluster
        cluster_points = part_positions[cluster_labels == cluster_id]
        plt.scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            color=custom_colors[cluster_id],  # Use predefined custom colors
            label=f"Cluster {cluster_id + 1}",
            s=20,  # Point size
            edgecolor="black",  # Edge color
            linewidth=0.5,  # Edge width
            zorder=2,  # Display above contours
        )

    # Set titles and labels
    plt.title(f"Fitness Landscape with {algorithm.capitalize()} Clustering (k={k})")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    # plt.legend() # Uncomment to show legend

    if save_fig:
        # Save fitness landscape plot
        landscape_path = os.path.join(
            output_dir, f"landscape/fitness_landscape_{algorithm}_k{k}.pdf"
        )
        plt.savefig(landscape_path, dpi=300, bbox_inches="tight")
        print(f"Saved fitness landscape to {landscape_path}")

    # plt.show() # Uncomment to display figure directly
    plt.close()

    return cluster_labels


def visualize_top_k_data_state_fitness_landscape(
    state_distance_matrix,
    log_distance_matrix,
    log_object,
    game_results,
    state_log,
    grid_x,
    grid_y,
    grid_z,
    topk=0.1,
    k_values=[3, 5, 7],
    algorithms=["kmeans", "spectral", "gmm", "agglomerative", "birch"],
    width=300,
    height=450,
    save_fig=False,
    enable_heatmap=False,
):
    # Calculate distances between states contained in the logs within the top-k fitness percentage,
    # as well as the distances between these log sequences.
    (
        top_k_state_distance_matrix,
        top_k_log_distance_matrix,
        top_k_indices,
        top_k_logs,
        top_k_states,
        top_k_positions,
        top_k_results,
    ) = get_top_k_solution_distances(
        state_distance_matrix,
        log_distance_matrix,
        log_object,
        game_results,
        state_log,
        topk=topk,
    )

    all_cluster_labels = {}

    # Create output directory
    output_dir = get_output_dir(OutputPaths.FITNESS_STANDARD)
    if save_fig and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Print experiment configuration
    print(f"Starting clustering analysis: algorithms {algorithms}, cluster counts {k_values}")
    print(f"Data scale: {len(top_k_positions)} samples, {top_k_positions.shape[1]} dimensional features")

    # Iterate through different algorithms and cluster counts
    for algorithm in algorithms:
        for k in k_values:
            print(f"\nProcessing: {algorithm} algorithm, k={k}")

            try:
                # Call clustering visualization function
                cluster_labels = visualize_part_distances(
                    top_k_log_distance_matrix,
                    top_k_positions,
                    top_k_results,
                    grid_x,
                    grid_y,
                    grid_z,
                    k=k,
                    algorithm=algorithm,
                    save_fig=save_fig,
                    enable_heatmap=enable_heatmap,
                )

                # Save clustering results
                key = f"{algorithm}_k{k}"
                all_cluster_labels[key] = cluster_labels

                # Print clustering statistics
                unique_labels, counts = np.unique(cluster_labels, return_counts=True)
                counts_py = [int(c) for c in counts]
                labels_py = [int(l) for l in unique_labels]
                print(f"Clustering results: {dict(zip(labels_py, counts_py))}")

            except Exception as e:
                print(f"Error: {algorithm} algorithm failed at k={k} - {str(e)}")
                continue

    # Plot state transition graph (using the last successful clustering result)
    # if all_cluster_labels:
    #     last_key = list(all_cluster_labels.keys())[-1]
    #     last_labels = all_cluster_labels[last_key]
    #     last_algorithm, last_k = last_key.split('_k')
    #     plot_state_transition_graph(top_k_state_distance_matrix, top_k_log_distance_matrix, top_k_logs, top_k_states,
    #                                 last_labels, k=int(last_k), width=width, height=height)

    print(f"\nClustering analysis complete, processed {len(all_cluster_labels)} configurations in total")
    print(f"Results are saved in the {output_dir} folder")

    return all_cluster_labels


def get_sequences_by_cluster(kmeans_labels, top_k_sequences):
    # Initialize a dictionary to store sequences for each cluster
    cluster_sequences = {}

    # Iterate through top_k_sequences and kmeans_labels
    for label, sequence in zip(kmeans_labels, top_k_sequences):
        if label not in cluster_sequences:
            cluster_sequences[label] = []
        cluster_sequences[label].append(sequence)

    return cluster_sequences


def plot_sankey_diagram(nodes, edges, node_to_cluster, custom_colors, k):
    """
    Plot Sankey Diagram
    :param nodes: Set of nodes
    :param edges: Set of edges, format: {(src, dst): value}
    :param node_to_cluster: Mapping from node to clusters, format: {node: [count1, count2, ..., countK]}
    :param custom_colors: Custom cluster colors
    :param k: Number of clusters
    """
    # Node indexing
    node_indices = {node: i for i, node in enumerate(nodes)}
    source_indices = [node_indices[src] for src, dst in edges.keys()]
    target_indices = [node_indices[dst] for src, dst in edges.keys()]
    values = list(edges.values())

    # Node colors (based on clustering)
    node_colors = []

    # Define nodes for Switch-Fire and Focus-Fire tactics
    fire_strategies = {
        "Switch-Fire Tactic": ["4b1b4b", "4c1c1c", "4b1b1b", "4d1d1d"],
        "Focus-Fire Tactic": ["4b4b4b", "4d4d4d", "4b4b4c", "4c4c4c"],
    }

    # Colors for custom strategy nodes
    strategy_colors = {
        "Switch-Fire Tactic": "#ef822f",
        "Focus-Fire Tactic": "#5d009a",
        "Unknown Tactic": "#808080",
    }

    # Custom starting node colors corresponding to each strategy
    strategy_node_colors = {
        "Switch-Fire Tactic": "#e39e33",
        "Focus-Fire Tactic": "#a631b4",
        "Unknown Tactic": "#C0C0C0",
    }

    # Modify node labels to only show labels for the first three and last columns
    node_labels = []

    incoming_nodes = {}
    for source, target in edges:
        if target not in incoming_nodes:
            incoming_nodes[target] = []
        incoming_nodes[target].append(source)

    # Calculate colors for each node
    for node in nodes:
        if "Cluster" in node:
            # Cluster label nodes use the corresponding cluster color
            cluster_index = int(node.split(" ")[1])
            node_colors.append(custom_colors[cluster_index])
            # node_labels.append(node)  # Cluster labels always display labels
            node_labels.append("")  # Cluster labels always display labels
        else:
            cluster_counts = node_to_cluster[node]
            total_count = sum(cluster_counts)
            if total_count == 0:
                node_colors.append("black")  # Use black if node doesn't appear in any cluster
                if "Switch-Fire Tactic" in node or "Focus-Fire Tactic" in node:
                    # Use custom strategy color if it's a strategy node
                    node_colors[-1] = strategy_colors.get(node, "black")
                    node_labels.append("")  # Strategy nodes always display labels
                    # node_labels.append(node)  # Strategy nodes always display labels
                else:
                    if node in fire_strategies["Switch-Fire Tactic"]:
                        # Use custom starting node color for Switch-Fire
                        node_colors[-1] = strategy_node_colors.get(
                            "Switch-Fire Tactic", "black"
                        )
                    elif node in fire_strategies["Focus-Fire Tactic"]:
                        node_colors[-1] = strategy_node_colors.get(
                            "Focus-Fire Tactic", "black"
                        )
                    # node_labels.append(node)  # Show label if forward node belongs to fire_strategies
                    node_labels.append(
                        ""
                    )  # Show label if forward node belongs to fire_strategies

            else:
                # Calculate the proportion of each cluster
                cluster_proportions = [count / total_count for count in cluster_counts]
                if max(cluster_proportions) == 1.0:
                    # If node appears in only one cluster, use that cluster's color directly
                    dominant_cluster = cluster_proportions.index(
                        max(cluster_proportions)
                    )
                    node_colors.append(custom_colors[dominant_cluster])
                else:
                    # Calculate mixed color based on cluster proportions
                    mixed_color = [0, 0, 0]  # RGB values
                    for i in range(k):
                        color = custom_colors[i].lstrip("#")
                        r, g, b = (
                            int(color[0:2], 16),
                            int(color[2:4], 16),
                            int(color[4:6], 16),
                        )
                        mixed_color[0] += r * cluster_proportions[i]
                        mixed_color[1] += g * cluster_proportions[i]
                        mixed_color[2] += b * cluster_proportions[i]
                    mixed_color = [int(c) for c in mixed_color]
                    mixed_color_hex = "#{:02x}{:02x}{:02x}".format(*mixed_color)
                    node_colors.append(mixed_color_hex)
                # Check if the forward node of this node belongs to fire_strategies
                if node in incoming_nodes and any(
                    incoming in fire_strategies["Switch-Fire Tactic"]
                    or incoming in fire_strategies["Focus-Fire Tactic"]
                    for incoming in incoming_nodes[node]
                ):
                    # node_labels.append(node)  # Show label if forward node belongs to fire_strategies
                    node_labels.append("")
                else:
                    node_labels.append("")  # Otherwise hide label

    # Plot Sankey Diagram
    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=5,
                    thickness=15,
                    line=dict(color="black", width=0.5),
                    label=node_labels,  # Use modified node labels
                    color=node_colors,
                ),
                link=dict(
                    source=source_indices,
                    target=target_indices,
                    value=values,
                ),
            )
        ]
    )

    fig.update_layout(title_text="", font_size=12)
    # Save as PDF file
    fig.write_image("sankey_diagram.pdf", width=800, height=1000, scale=4)
    fig.show()


def visualize_top_k_data_pattern(
    state_distance_matrix,
    log_distance_matrix,
    log_object,
    game_results,
    state_log,
    grid_x,
    grid_y,
    grid_z,
    topk=0.1,
    k=3,
    width=300,
    height=450,
):
    # Calculate distances between states in the top-k fitness logs,
    # and distances between these log sequences.
    (
        top_k_state_distance_matrix,
        top_k_log_distance_matrix,
        top_k_indices,
        top_k_logs,
        top_k_states,
        top_k_positions,
        top_k_results,
    ) = get_top_k_solution_distances(
        state_distance_matrix,
        log_distance_matrix,
        log_object,
        game_results,
        state_log,
        topk=topk,
    )

    # Calculate K-Means clustering labels
    kmeans_labels = kmeans_clustering(
        top_k_log_distance_matrix, top_k_positions, top_k_results, k=k
    )

    log_path = action_log_path
    result_path = game_result_path
    all_sequences = get_marked_sequences(log_path, result_path, min_support=0.01)

    top_k_sequences = [
        (all_sequences[i][1], all_sequences[i][2]) for i in top_k_indices
    ]

    cluster_sequences = get_sequences_by_cluster(
        kmeans_labels[-100:], top_k_sequences[-100:]
    )

    # Define nodes for Switch-Fire and Focus-Fire tactics
    fire_strategies = {
        "Switch-Fire Tactic": ["4b1b4b", "4c1c1c", "4b1b1b", "4d1d1d"],
        "Focus-Fire Tactic": ["4b4b4b", "4d4d4d", "4b4b4c", "4c4c4c"],
    }

    # Extract nodes and edges
    nodes = set()
    edges = defaultdict(int)
    node_to_cluster = defaultdict(
        lambda: [0] * k
    )  # Used to track occurrence count of each node in each cluster

    # Extract parts wrapped in brackets 【】, and take the first 6 chars inside the first 【】 as the start node
    def extract_nodes(sequence: str):
        """
        Returns:
            nodes   -> List of all 【...】 original strings
            first   -> First 6 characters of text inside the first 【】 (or all if shorter)
        """
        nodes = []
        first = None
        start = 0
        while True:
            start = sequence.find("【", start)
            if start == -1:
                break
            end = sequence.find("】", start)
            if end == -1:  # Break if no closing bracket
                break
            # Save the node with brackets intact
            nodes.append(sequence[start : end + 1])
            # On the first occurrence, extract first 6 characters of the inner text
            if first is None:
                inner = sequence[start + 1 : end]  # Remove 【】
                first = inner[:6]  # Max 6 characters
            start = end + 1  # Continue searching
        return nodes, first

    # Classify start node into Switch-Fire or Focus-Fire tactics
    def classify_start_node(start_node):
        for strategy, patterns in fire_strategies.items():
            if start_node in patterns:
                return strategy
        return "Unknown Tactic"

    # Collect all scores for normalization
    all_scores = []
    for cluster, samples in cluster_sequences.items():
        for sequence, score in samples:
            all_scores.append(score)

    if all_scores:
        min_score = min(all_scores)
        max_score = max(all_scores)

        # Avoid division by zero if all scores are equal
        if max_score == min_score:

            def normalize_score(score):
                return 1.0
        else:

            def normalize_score(score):
                # Normalize to 0-1 range
                return (score - min_score) / (max_score - min_score)
    else:

        def normalize_score(score):
            return 1.0

    # Function for cycle detection and edge weight accumulation
    def add_edge_without_cycles(source, target, weight, edges_dict):
        """
        Add edge weight, filtering out edges that create cycles (including A->B->C->A types)
        :param source: Source node
        :param target: Target node
        :param weight: Weight
        :param edges_dict: Edge dictionary
        """
        # Check for direct reverse edge
        reverse_key = (target, source)
        if reverse_key in edges_dict:
            return  # Direct reverse edge exists, skip adding

        # Check if adding this edge creates a cycle
        # Use DFS to check if source is reachable from target
        if would_create_cycle(source, target, edges_dict):
            return  # Would create cycle, skip adding

        # No cycle created, add normally
        if (source, target) in edges_dict:
            edges_dict[(source, target)] += weight
        else:
            edges_dict[(source, target)] = weight

    def would_create_cycle(source, target, edges_dict):
        """
        Check if adding an edge from source to target creates a cycle
        i.e., check if source is reachable from target
        """
        visited = set()
        stack = [target]

        while stack:
            current = stack.pop()
            if current == source:
                return True  # Path found, cycle exists

            if current in visited:
                continue
            visited.add(current)

            # Find all edges starting from current
            for src, dst in edges_dict.keys():
                if src == current and dst not in visited:
                    stack.append(dst)

        return False  # No path found, no cycle created

    # Extract data
    for cluster, samples in cluster_sequences.items():
        for sequence, score in samples:
            normalized_score = normalize_score(score)
            parts, first_node = extract_nodes(sequence)
            if first_node:  # If start node exists
                classified_strategy = classify_start_node(
                    first_node
                )  # Classify into Switch-Fire or Focus-Fire
                nodes.add(classified_strategy)  # Add strategy node to set
                nodes.add(first_node)  # Add start node to set
                add_edge_without_cycles(
                    classified_strategy, first_node, normalized_score, edges
                )  # Add edge from strategy node to start node
                if parts:  # If other nodes exist
                    add_edge_without_cycles(
                        first_node, parts[0], normalized_score, edges
                    )  # Add edge from start node to first sequence node
            for i in range(len(parts) - 1):
                src = parts[i]
                dst = parts[i + 1]
                nodes.add(src)
                nodes.add(dst)
                add_edge_without_cycles(src, dst, normalized_score, edges)
                node_to_cluster[src][cluster] += normalized_score
                node_to_cluster[dst][cluster] += normalized_score
            # Add edge from the last node to the cluster label
            if parts:
                last_node = parts[-1]
                add_edge_without_cycles(
                    last_node, f"Cluster {cluster}", normalized_score, edges
                )

    # Add cluster labels as final nodes
    for cluster in cluster_sequences.keys():
        nodes.add(f"Cluster {cluster}")

    # Call plotting function
    plot_sankey_diagram(nodes, edges, node_to_cluster, custom_colors, k)


def generate_dynamic_colors(k):
    """
    Generate dynamic color scheme
    :param k: Number of colors required
    :return: List of colors
    """
    if k <= 8:
        # Use existing custom colors
        return custom_colors[:k]

    # Generate more colors using ColorBrewer's Set1 color scheme
    base_colors = [
        "#e41a1c",  # Red
        "#377eb8",  # Blue
        "#4daf4a",  # Green
        "#984ea3",  # Purple
        "#ff7f00",  # Orange
        "#ffff33",  # Yellow
        "#a65628",  # Brown
        "#f781bf",  # Pink
        "#999999",  # Grey
        "#1b9e77",  # Teal
        "#d95f02",  # Dark Orange
        "#7570b3",  # Dark Purple
        "#e7298a",  # Magenta
        "#66a61e",  # Grass Green
        "#e6ab02",  # Gold
    ]

    # If more colors are needed, generate by adjusting hue
    if k <= len(base_colors):
        return base_colors[:k]
    else:
        # Generate additional colors
        additional_colors = []
        for i in range(k - len(base_colors)):
            hue = (i * 360 // (k - len(base_colors))) % 360
            # Use HSL to generate colors, keeping medium saturation and brightness
            import colorsys

            rgb = colorsys.hsv_to_rgb(hue / 360, 0.7, 0.8)
            hex_color = "#{:02x}{:02x}{:02x}".format(
                int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
            )
            additional_colors.append(hex_color)

        return base_colors + additional_colors


def determine_optimal_clusters(distance_matrix, max_clusters=50, min_clusters=2):
    """
    Determine the optimal number of clusters using hierarchical clustering and silhouette coefficient
    :param distance_matrix: Distance matrix
    :param max_clusters: Maximum number of clusters
    :param min_clusters: Minimum number of clusters
    :return: Optimal number of clusters
    """
    from sklearn.metrics import silhouette_score
    from scipy.cluster.hierarchy import fcluster

    # Convert distance matrix to condensed form
    condensed_distance_matrix = squareform(distance_matrix)

    # Compute hierarchical clustering linkage matrix
    linkage_matrix = linkage(condensed_distance_matrix, method="average")

    best_score = -1
    best_k = min_clusters

    # Test different cluster counts
    for k in range(min_clusters, min(max_clusters + 1, len(distance_matrix))):
        try:
            # Obtain labels from hierarchical clustering
            cluster_labels = fcluster(linkage_matrix, k, criterion="maxclust")

            # Calculate silhouette score
            if len(np.unique(cluster_labels)) > 1:  # Ensure multiple clusters exist
                score = silhouette_score(
                    distance_matrix, cluster_labels - 1, metric="precomputed"
                )

                if score > best_score:
                    best_score = score
                    best_k = k

        except Exception as e:
            print(f"Error calculating silhouette score for k={k}: {e}")
            continue

    print(
        f"Optimal number of clusters determined: {best_k} (silhouette score: {best_score:.3f})"
    )
    return best_k


def hierarchical_clustering_analysis(distance_matrix, k=None):
    """
    Perform hierarchical clustering analysis
    :param distance_matrix: Distance matrix
    :param k: Specified number of clusters; automatically determined if None
    :return: Cluster labels, actual number of clusters used, linkage matrix
    """
    # Convert distance matrix to condensed form
    condensed_distance_matrix = squareform(distance_matrix)

    # Compute hierarchical clustering linkage matrix
    linkage_matrix = linkage(condensed_distance_matrix, method="average")

    # If k is not specified, automatically determine the optimal number of clusters
    if k is None:
        k = determine_optimal_clusters(distance_matrix)

    # Obtain labels from hierarchical clustering
    cluster_labels = fcluster(linkage_matrix, k, criterion="maxclust")

    # Convert to 0-based indexing
    cluster_labels = cluster_labels - 1

    return cluster_labels, k, linkage_matrix


def visualize_top_k_data_pattern_comprehensive(
    state_distance_matrix,
    log_distance_matrix,
    log_object,
    game_results,
    state_log,
    topk=0.1,
    max_clusters=10,
    min_clusters=2,
    width=300,
    height=450,
    map_id="map1",
    data_id="data1",
    show_labels=True,
):
    """
    Comprehensive visualization function: Automatically determines the number of clusters
    and analyzes patterns in TopK data.
    :param state_distance_matrix: Matrix of distances between states
    :param log_distance_matrix: Matrix of distances between logs
    :param log_object: List of fitness values
    :param game_results: List of game results
    :param state_log: Sequences of state logs
    :param topk: Percentage of data with the highest fitness values to select
    :param max_clusters: Limit for the maximum number of clusters
    :param min_clusters: Limit for the minimum number of clusters
    :param width: Image width
    :param height: Image height
    """

    print(
        "Starting comprehensive pattern analysis with automatic cluster determination..."
    )

    # 1. Extract TopK data
    print(f"Extracting top {topk * 100}% fitness data...")
    (
        top_k_state_distance_matrix,
        top_k_log_distance_matrix,
        top_k_indices,
        top_k_logs,
        top_k_states,
        top_k_positions,
        top_k_results,
    ) = get_top_k_solution_distances(
        state_distance_matrix,
        log_distance_matrix,
        log_object,
        game_results,
        state_log,
        topk=topk,
    )

    print(f"Processing {len(top_k_results)} sequences...")

    # 2. Automatically determine the optimal number of clusters
    print("Determining optimal number of clusters using hierarchical clustering...")
    kmeans_labels, optimal_k, linkage_matrix = hierarchical_clustering_analysis(
        top_k_log_distance_matrix, k=None
    )

    print(f"Automatically determined optimal k = {optimal_k}")

    # 3. Generate dynamic color scheme
    dynamic_colors = generate_dynamic_colors(optimal_k)
    print(f"Generated {len(dynamic_colors)} colors for visualization")

    # 4. Visualize clustering results and fitness landscape (if needed)
    # Fitness landscape visualization can be added here;
    # currently, the primary focus is the Sankey diagram.

    # 5. Pattern analysis and Sankey diagram generation
    log_path = action_log_path
    result_path = game_result_path
    all_sequences = get_marked_sequences(log_path, result_path, min_support=0.01)

    top_k_sequences = [
        (all_sequences[i][1], all_sequences[i][2]) for i in top_k_indices
    ]

    # Group clustering sequences using the actual data length
    cluster_sequences = get_sequences_by_cluster(kmeans_labels, top_k_sequences)

    # Extract nodes and edges
    nodes = set()
    edges = defaultdict(int)
    node_to_cluster = defaultdict(lambda: [0] * optimal_k)

    # Extract parts wrapped in brackets 【】, and take the first 6 chars inside the first 【】 as the start node
    def extract_nodes(sequence: str):
        """
        Returns:
            nodes   -> List of all 【...】 original strings
            first   -> First 6 characters of text inside the first 【】 (or all if shorter)
        """
        nodes = []
        first = None
        start = 0
        while True:
            start = sequence.find("【", start)
            if start == -1:
                break
            end = sequence.find("】", start)
            if end == -1:  # Break if no closing bracket
                break
            # Save the node with brackets intact
            nodes.append(sequence[start : end + 1])
            # On the first occurrence, extract first 6 characters of the inner text
            if first is None:
                inner = sequence[start + 1 : end]  # Remove 【】
                first = inner[:6]  # Max 6 characters
            start = end + 1  # Continue searching
        return nodes, first

    action_dict = create_action_dictionary()

    # Use the new intelligent strategy node analysis function
    def classify_start_node(start_node):
        strategy_labels = analyze_strategy_node(start_node, action_dict)
        return strategy_labels

    # Collect all scores for normalization
    print("Collecting scores for normalization...")
    all_scores = []
    for cluster, samples in cluster_sequences.items():
        for sequence, score in samples:
            all_scores.append(score)

    if all_scores:
        min_score = min(all_scores)
        max_score = max(all_scores)
        print(f"Score range: [{min_score:.3f}, {max_score:.3f}]")

        # Avoid division by zero if all scores are equal
        if max_score == min_score:
            print("All scores are equal, using uniform weights")

            def normalize_score(score):
                return 1.0
        else:

            def normalize_score(score):
                # Normalize to 0-1 range
                return (score - min_score) / (max_score - min_score)
    else:
        print("No scores found, using uniform weights")

        def normalize_score(score):
            return 1.0

    # Cycle detection and edge weight accumulation function
    def add_edge_without_cycles(source, target, weight, edges_dict):
        """
        Add edge weight, filtering out edges that create cycles (including A->B->C->A types)
        :param source: Source node
        :param target: Target node
        :param weight: Weight
        :param edges_dict: Edge dictionary
        """
        # Check for direct reverse edge
        reverse_key = (target, source)
        if reverse_key in edges_dict:
            return  # Direct reverse edge exists, skip adding

        # Check if adding this edge creates a cycle
        # Use DFS to check if source is reachable from target
        if would_create_cycle(source, target, edges_dict):
            return  # Would create cycle, skip adding

        # No cycle created, add normally
        if (source, target) in edges_dict:
            edges_dict[(source, target)] += weight
        else:
            edges_dict[(source, target)] = weight

    def would_create_cycle(source, target, edges_dict):
        """
        Check if adding an edge from source to target creates a cycle
        i.e., check if source is reachable from target
        """
        visited = set()
        stack = [target]

        while stack:
            current = stack.pop()
            if current == source:
                return True  # Path found, cycle exists

            if current in visited:
                continue
            visited.add(current)

            # Find all edges starting from current
            for src, dst in edges_dict.keys():
                if src == current and dst not in visited:
                    stack.append(dst)

        return False  # No path found, no cycle created

    # Extract data
    print("Extracting behavioral patterns...")
    total_sequences = 0
    for cluster, samples in cluster_sequences.items():
        print(f"  Cluster {cluster}: {len(samples)} sequences")
        total_sequences += len(samples)
        for sequence, score in samples:
            normalized_score = normalize_score(score)
            parts, first_node = extract_nodes(sequence)
            if first_node:
                classified_strategy = classify_start_node(first_node)
                nodes.add(classified_strategy)
                nodes.add(first_node)
                add_edge_without_cycles(
                    classified_strategy, first_node, normalized_score, edges
                )
                if parts:
                    add_edge_without_cycles(
                        first_node, parts[0], normalized_score, edges
                    )
            for i in range(len(parts) - 1):
                src = parts[i]
                dst = parts[i + 1]
                nodes.add(src)
                nodes.add(dst)
                add_edge_without_cycles(src, dst, normalized_score, edges)
                node_to_cluster[src][cluster] += normalized_score
                node_to_cluster[dst][cluster] += normalized_score
            if parts:
                last_node = parts[-1]
                add_edge_without_cycles(
                    last_node, f"Cluster {cluster}", normalized_score, edges
                )

    print(f"Processed {total_sequences} total sequences across {optimal_k} clusters")

    # Add cluster labels as final nodes
    for cluster in cluster_sequences.keys():
        nodes.add(f"Cluster {cluster}")

    # Call plotting function using dynamically generated colors
    print("Generating Sankey diagram...")
    plot_sankey_diagram_adaptive(
        nodes,
        edges,
        node_to_cluster,
        dynamic_colors,
        optimal_k,
        map_id,
        data_id,
        show_labels=show_labels,
    )

    print(f"Comprehensive pattern analysis completed with {optimal_k} clusters")

    return {
        "optimal_k": optimal_k,
        "cluster_labels": kmeans_labels,
        "dynamic_colors": dynamic_colors,
        "cluster_sequences": cluster_sequences,
        "linkage_matrix": linkage_matrix,
    }


def create_action_dictionary():
    """
    Dynamically creates an action dictionary by reading CSV files from the action_path
    directory to obtain real action names.
    Key is a letter starting from 'a', value is the corresponding action name.
    """
    import os
    import pandas as pd

    try:
        # Get list of CSV files in the action_path directory
        if os.path.exists(action_path):
            csv_files = [f for f in os.listdir(action_path) if f.endswith(".csv")]
            if csv_files:
                # Read the first CSV file
                first_csv = csv_files[0]
                csv_path = os.path.join(action_path, first_csv)

                # Read the CSV
                df = pd.read_csv(csv_path)

                # Get all action columns (excluding Unnamed columns)
                action_columns = [col for col in df.columns if col != "Unnamed: 0"]

                # Create mapping from a, b, c... to action names
                action_dict = {}
                for i, action_name in enumerate(action_columns):
                    key = chr(ord("a") + i)
                    action_dict[key] = action_name

                print(f"Loaded {len(action_dict)} actions from CSV: {first_csv}")
                return action_dict

        # Fallback to default dictionary if CSV cannot be read
        print(
            "Warning: Could not load actions from CSV, using default action dictionary"
        )
        default_action_dict = {
            "a": "action_ATK_nearest",
            "b": "action_ATK_clu_nearest",
            "c": "action_ATK_nearest_weakest",
            "d": "action_ATK_clu_nearest_weakest",
            "e": "action_ATK_threatening",
            "f": "action_DEF_clu_nearest",
            "g": "action_MIX_gather",
            "h": "action_MIX_lure",
            "i": "action_MIX_sacrifice_lure",
            "j": "do_randomly",
            "k": "do_nothing",
        }
        return default_action_dict

    except Exception as e:
        print(f"Error loading action dictionary: {e}")
        print("Using default action dictionary")
        default_action_dict = {
            "a": "action_ATK_nearest",
            "b": "action_ATK_clu_nearest",
            "c": "action_ATK_nearest_weakest",
            "d": "action_ATK_clu_nearest_weakest",
            "e": "action_ATK_threatening",
            "f": "action_DEF_clu_nearest",
            "g": "action_MIX_gather",
            "h": "action_MIX_lure",
            "i": "action_MIX_sacrifice_lure",
            "j": "do_randomly",
            "k": "do_nothing",
        }
        return default_action_dict


def analyze_strategy_node(first_node, action_dict):
    """
    Analyzes the 6-character start node sequence and generates a multi-label string.

    Args:
        first_node (str): 6-character start sequence, e.g., "4b4b4b"

    Returns:
        str: Strategy label string, e.g., "'sustained coordination''greedy attack'"
    """
    if not first_node:
        return "'Unknown Tactic'"

    # Split into 3 action groups (2 chars each): 1st char is cluster intensity, 2nd is action key
    action_groups = []
    cluster_intensities = []

    for i in range(0, len(first_node) // 2 * 2, 2):
        cluster_intensity = int(first_node[i])  # Intensity 0-4
        action_key = first_node[i + 1]  # Key a-k
        action_name = action_dict.get(action_key, "unknown_action")

        action_groups.append(action_name)
        cluster_intensities.append(cluster_intensity)

    labels = []

    # Level 1: Analyze strategy rules for each action group
    attack_count = 0
    defense_count = 0
    mix_count = 0
    inefficient_count = 0

    for action_name in action_groups:
        # Analyze action type
        if "ATK" in action_name:
            attack_count += 1
            if action_name in [
                "action_ATK_nearest",
                "action_ATK_clu_nearest",
                "action_ATK_nearest_weakest",
                "action_ATK_clu_nearest_weakest",
            ]:
                labels.append("greedy")
            elif action_name == "action_ATK_threatening":
                labels.append("threat-focused")
        elif "DEF" in action_name:
            defense_count += 1
            if action_name == "action_DEF_clu_nearest":
                labels.append("exfiltration")
        elif "MIX" in action_name:
            mix_count += 1
            if action_name == "action_MIX_gather":
                labels.append("massing")
            elif action_name == "action_MIX_lure":
                labels.append("feint")
            elif action_name == "action_MIX_sacrifice_lure":
                labels.append("sacrificial feint")
        elif action_name in ["do_randomly", "do_nothing"]:
            labels.append("greedy")
        elif action_name in ["do_nothing"]:
            inefficient_count += 1
            labels.append("inefficiency")

    # Analyze cluster intensity
    unique_intensities = set(cluster_intensities)
    for intensity in unique_intensities:
        if intensity == 0:
            labels.append("independently")
        elif 1 <= intensity <= 3:
            labels.append("distributionally")
        elif intensity == 4:
            labels.append("centrally")

    # Level 2: Analyze coordination patterns
    if len(unique_intensities) == 1:
        # Check for exceptions (ATK_nearest or ATK_nearest_weakest)
        has_exception = False
        for i, action_name in enumerate(action_groups):
            if action_name in ["action_ATK_nearest", "action_ATK_nearest_weakest"]:
                has_exception = True
                break

        if not has_exception:
            labels.append("sustained coordination")
        else:
            labels.append("switched coordination")  # Exception found, so coordination switched
    else:
        labels.append("switched coordination")

    # Analyze tactical patterns
    if attack_count == 3:
        labels.append("sustained attack")
    elif defense_count == 3:
        labels.append("sustained defense")
    elif mix_count == 3:
        labels.append("sustained hybrid")
    elif attack_count > 0 and defense_count > 0:
        labels.append("switched tactic")
    elif attack_count > 0 and mix_count > 0:
        labels.append("switched tactic")
    elif defense_count > 0 and mix_count > 0:
        labels.append("switched tactic")
    elif inefficient_count >= 2:
        labels.append("inefficient tactic")

    # Remove duplicates
    labels = list(set(labels))

    # Return unknown tactic if no labels generated
    if not labels or labels == ["Unknown Tactic"]:
        return "'Unknown Tactic'"

    # Sort and convert to string format
    labels = sorted(list(set(labels)))
    result = ""
    for label in labels:
        result += f"'{label}'"

    return result


def visualize_specific_pattern_comprehensive(
    all_cluster_results,
    state_distance_matrix,
    log_distance_matrix,
    log_object,
    game_results,
    state_log,
    seed=42,
    enable_sampling=True,
    sampling_ratio=0.5,
    sampling_mode="completely_random",
    width=300,
    height=450,
    map_id="map1",
    data_id="data1",
    show_labels=True,
    doublerow_annotation=False,
):
    """
    Function to draw a Sankey diagram based on existing clustering results.
    :param all_cluster_results: A dictionary containing all clustering results, where each element
                                includes algorithm, k, labels, etc.
    :param state_distance_matrix: State distance matrix.
    :param log_distance_matrix: Log distance matrix.
    :param log_object: List of fitness values.
    :param game_results: List of game results.
    :param state_log: State log sequences.
    :param enable_sampling: Whether to sample from each cluster's instances.
    :param sampling_ratio: Sampling ratio (0.0-1.0).
    :param sampling_mode: Sampling mode, options:
                         - "completely_random": Pure random sampling.
                         - "top_random": Sort by score descending, take the top sampling_ratio high-score samples.
                         - "back_random": Sort by score ascending, take the top sampling_ratio low-score samples.
    :param width: Image width.
    :param height: Image height.
    :param map_id: Map ID.
    :param data_id: Data ID.
    :param show_labels: Whether to display labels.
    :return: Processing result statistics.
    """

    print("Starting pattern analysis based on existing clustering results...")

    # Create output directory
    output_dir = "output_sankey"
    import os
    from collections import defaultdict

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    action_dict = create_action_dictionary()

    def classify_start_node(start_node):
        strategy_labels = analyze_strategy_node(start_node, action_dict)
        return strategy_labels

    def classify_part_node(part_node):
        clean = re.sub(r"[【】]", "", part_node)
        strategy_labels = analyze_strategy_node(clean, action_dict)
        return strategy_labels

    # Get all sequence data
    log_path = action_log_path
    result_path = game_result_path
    all_sequences = get_marked_sequences(log_path, result_path, min_support=0.01)

    # Statistics
    total_diagrams = 0
    successful_diagrams = 0
    results_summary = []

    # Iterate through all clustering results
    for algorithm, cluster_labels in all_cluster_results.items():
        # 2. Determine optimal k and data cluster labels
        k = int(algorithm.split("_")[-1].replace("k", ""))
        labels = cluster_labels

        print(f"\n--- Processing {algorithm.upper()} with k={k} ---")

        # 3. Generate dynamic color scheme
        dynamic_colors = generate_dynamic_colors(k)

        if enable_sampling:
            # Multi-mode sampling logic
            print(
                f"  Sampling {sampling_ratio * 100:.0f}% of sequences from each cluster (mode: {sampling_mode})..."
            )
            sampled_indices = []

            np.random.seed(42)

            # Get indices and corresponding scores for each cluster
            for cluster_id in range(k):
                cluster_indices = [
                    i for i, label in enumerate(labels) if label == cluster_id
                ]
                if cluster_indices:
                    # Get scores for these indices
                    cluster_scores = [log_object[i] for i in cluster_indices]
                    cluster_data = list(zip(cluster_indices, cluster_scores))

                    # Calculate number of samples needed
                    n_sample = max(1, int(len(cluster_indices) * sampling_ratio))

                    # Sample based on the selected mode
                    if sampling_mode == "completely_random":
                        # Pure random sampling
                        sampled_cluster_indices = np.random.choice(
                            cluster_indices, n_sample, replace=False
                        )
                        sampled_indices.extend(sampled_cluster_indices)

                    elif sampling_mode == "top_random":
                        # Sort by score descending, take first n_sample
                        cluster_data.sort(key=lambda x: x[1], reverse=True)
                        sampled_cluster_indices = [
                            data[0] for data in cluster_data[:n_sample]
                        ]
                        sampled_indices.extend(sampled_cluster_indices)

                    elif sampling_mode == "back_random":
                        # Sort by score ascending, take first n_sample
                        cluster_data.sort(key=lambda x: x[1])
                        sampled_cluster_indices = [
                            data[0] for data in cluster_data[:n_sample]
                        ]
                        sampled_indices.extend(sampled_cluster_indices)

                    else:
                        print(
                            f"  Warning: Unknown sampling mode '{sampling_mode}', using completely_random"
                        )
                        sampled_cluster_indices = np.random.choice(
                            cluster_indices, n_sample, replace=False
                        )
                        sampled_indices.extend(sampled_cluster_indices)

            # Build new sequence lists based on sampled indices
            sampled_sequences = [
                (all_sequences[i][1], all_sequences[i][2]) for i in sampled_indices
            ]
            sampled_labels = [labels[i] for i in sampled_indices]
            sampled_results = [all_sequences[i][2] for i in sampled_indices]

            # Group using sampled data
            cluster_sequences = get_sequences_by_cluster(
                sampled_labels, sampled_sequences
            )
            print(
                f"  Sampled {len(sampled_sequences)} sequences from {len(labels)} total sequences"
            )
        else:
            # Use all data
            sequences = [
                (all_sequences[i][1], all_sequences[i][2]) for i in range(len(labels))
            ]
            cluster_sequences = get_sequences_by_cluster(labels, sequences)
            print(f"  Using all {len(sequences)} sequences")
            sampled_labels = [labels[i] for i in sequences]
            sampled_results = [all_sequences[i][2] for i in sampled_labels]

        info = defaultdict(lambda: {"count": 0, "values": []})

        for label, value in zip(sampled_labels, sampled_results):
            info[label]["count"] += 1
            info[label]["values"].append(value)

        # Calculate statistics
        summary = {}
        for label, data in info.items():
            vals = data["values"]
            summary[label] = {
                "count": data["count"],
                "max": max(vals),
                "min": min(vals),
                "mean": round(sum(vals) / len(vals), 2),
            }

        # Extract nodes and edges
        nodes = set()
        edges = defaultdict(int)
        node_to_cluster = defaultdict(lambda: [0] * k)

        # Extract parts wrapped in brackets 【】 and extract the first 6 chars of the first 【】 as the start node
        def extract_nodes(sequence: str):
            """
            Returns:
                nodes   -> List of all 【...】 original strings.
                first   -> First 6 characters of text inside the first 【】 (or all if shorter).
            """
            nodes = []
            first = None
            start = 0
            while True:
                start = sequence.find("【", start)
                if start == -1:
                    break
                end = sequence.find("】", start)
                if end == -1:  # Break if no closing bracket
                    break
                # Save the node with brackets intact
                nodes.append(sequence[start : end + 1])
                # On the first occurrence, extract first 6 characters of the inner text
                if first is None:
                    inner = sequence[start + 1 : end]  # Remove 【】
                    first = inner[:6]  # Max 6 characters
                start = end + 1  # Continue searching
            return nodes, first

        # Collect all scores for normalization
        print("Collecting scores for normalization...")
        all_scores = []
        for cluster, samples in cluster_sequences.items():
            for sequence, score in samples:
                all_scores.append(score)

        if all_scores:
            min_score = min(all_scores)
            max_score = max(all_scores)
            print(f"Score range: [{min_score:.3f}, {max_score:.3f}]")

            # Avoid division by zero if all scores are equal
            if max_score == min_score:
                print("All scores are equal, using uniform weights")

                def normalize_score(score):
                    return 1.0
            else:

                def normalize_score(score):
                    # Normalize to 0-1 range
                    return (score - min_score) / (max_score - min_score)
        else:
            print("No scores found, using uniform weights")

            def normalize_score(score):
                return 1.0

        def add_edge_allowing_cycle(source, target, weight, edges_dict):
            """
            Only prevents the creation of reverse edges (target -> source);
            Forward edges (source -> target) are accumulated regardless,
            even if they create a cycle like A -> B -> C -> A.
            """
            # 1. If reverse edge already exists, return without adding
            if (target, source) in edges_dict:
                edges_dict[(source, target)] += weight
                return

            # 2. Otherwise, accumulate forward edge unconditionally
            edges_dict[(source, target)] += weight

        # Cycle detection and edge weight accumulation function
        def add_edge_without_cycles(source, target, weight, edges_dict):
            """
            Add edge weights, filtering out edges that create cycles (including A->B->C->A types)
            :param source: Source node
            :param target: Target node
            :param weight: Weight
            :param edges_dict: Edges dictionary
            """
            # Check for direct reverse edge
            reverse_key = (target, source)
            if reverse_key in edges_dict:
                return  # Direct reverse edge exists, skip adding

            # Check if adding this edge creates a cycle
            # Use DFS to check if source is reachable from target
            if would_create_cycle(source, target, edges_dict):
                return  # Would create cycle, skip adding

            # No cycle created, add normally
            if (source, target) in edges_dict:
                edges_dict[(source, target)] += weight
            else:
                edges_dict[(source, target)] = weight

        def would_create_cycle(source, target, edges_dict):
            """
            Check if adding an edge from source to target creates a cycle.
            i.e., check if source is reachable from target.
            """
            visited = set()
            stack = [target]

            while stack:
                current = stack.pop()
                if current == source:
                    return True  # Path found, cycle created

                if current in visited:
                    continue
                visited.add(current)

                # Find all edges starting from current
                for src, dst in edges_dict.keys():
                    if src == current and dst not in visited:
                        stack.append(dst)

            return False  # No path found, no cycle created

        def process_patterns_lists(
            indexed_tactics_lists, node_to_cluster, color_mode="fresh"
        ):
            edges = defaultdict(int)
            all_renamed_nodes = []
            temp_nodes = []
            temp_edges = defaultdict(int)
            for sublist, cluster_id in indexed_tactics_lists:
                # Independent numbering within the current list
                local_counter = 1
                renamed_sequence = []
                for tactic in sublist:
                    renamed = f"{tactic}({local_counter})"
                    local_counter += 1
                    renamed_sequence.append(renamed)
                    all_renamed_nodes.append(renamed)
                    if color_mode == "fresh":
                        node_to_cluster[renamed][cluster_id] += 1
                    else:
                        node_to_cluster[tactic][cluster_id] += 1
                temp_nodes.append(f"Cluster {cluster_id}")

                # Extract edges
                end = None
                for i in range(len(renamed_sequence) - 1):
                    src = renamed_sequence[i]
                    dst = renamed_sequence[i + 1]
                    edges[(src, dst)] += 1
                    end = dst
                temp_edges[(end, f"Cluster {cluster_id}")] += 1

            all_renamed_nodes.extend(temp_nodes)
            edges.update(temp_edges)
            # Deduplicate while maintaining order
            seen = set()
            unique_nodes = []
            for node in all_renamed_nodes:
                if node not in seen:
                    seen.add(node)
                    unique_nodes.append(node)

            return set(unique_nodes), dict(edges)

        # Extract data
        print("Extracting behavioral patterns...")
        total_sequences = 0
        all_patterns = set()
        raw_pattern_lists = []
        for cluster, samples in cluster_sequences.items():
            print(f"  Cluster {cluster}: {len(samples)} sequences")
            total_sequences += len(samples)
            for sequence, score in samples:
                parts, first_node = extract_nodes(sequence)
                open_tactic = classify_start_node(first_node)
                first_pattern = first_node
                parts_patterns = [part for part in parts]
                pattern_list = [open_tactic] + [first_pattern] + parts_patterns
                if first_node and parts:
                    raw_pattern_lists.append((pattern_list, cluster))
                    all_patterns.update(pattern_list)

        # Establish global mapping
        unique_patterns = list(OrderedDict.fromkeys(all_patterns))  # Maintain insertion order
        pattern_dict = {}
        counter = 1
        for pat in unique_patterns:
            if (
                "【" in pat and "】" in pat
            ):  # Also could use pat.startswith('【') and pat.endswith('】')
                pattern_dict[pat] = f"P{counter}"
                counter += 1
            else:
                pattern_dict[pat] = pat  # Keep as is

        # ---------- Step 2: Uniform Indexing ----------
        indexed_tactics_lists = [
            ([pattern_dict[t] for t in tactic_list], cluster)
            for tactic_list, cluster in raw_pattern_lists
        ]

        color_mode = "fresh"  # Options: unite, fresh
        nodes, edges = process_patterns_lists(
            indexed_tactics_lists, node_to_cluster, color_mode
        )

        removed_counter = Counter()
        print(f"Processed {total_sequences} total sequences across {k} clusters")

        # Add cluster labels as final nodes
        for cluster in cluster_sequences.keys():
            nodes.add(f"Cluster {cluster}")

        # Get labels for the current clustering algorithm
        current_key = f"{algorithm}"
        current_cluster_labels = None

        if current_key in all_cluster_results:
            current_cluster_labels = all_cluster_results[current_key]

        # Call plotting function with dynamically generated colors
        print("Generating Sankey diagram...")
        plot_sankey_diagram_adaptive(
            nodes,
            edges,
            node_to_cluster,
            dynamic_colors,
            k,
            map_id,
            data_id,
            sampling_mode,
            algorithm,
            show_labels=show_labels,
            summary=summary,
            removed_counter=removed_counter,
            doublerow_annotation=doublerow_annotation,
        )

        print(f"Comprehensive pattern analysis completed with {k} clusters")

        total_diagrams += 1

    # Print summary
    print(f"\n{'=' * 60}")
    print("SANKEY DIAGRAM GENERATION SUMMARY")
    print("=" * 60)
    print(f"Total processing attempts: {total_diagrams}")
    print(f"Successful diagrams: {successful_diagrams}")
    print(f"Success rate: {successful_diagrams / total_diagrams * 100:.1f}%")

    print(f"\nDetailed Results:")
    print(
        f"{'Algorithm':<15} {'K':<5} {'Clusters':<10} {'Sequences':<10} {'Status':<15} {'Output File'}"
    )
    print("-" * 80)

    for result in results_summary:
        if result["status"] == "success":
            print(
                f"{result['algorithm']:<15} {result['k']:<5} {result['clusters']:<10} {result['sequences']:<10} {result['status']:<15} {result['output_file']}"
            )
        elif result["status"] == "error":
            print(
                f"{result['algorithm']:<15} {result['k']:<5} {'-':<10} {'-':<10} {result['status']:<15} Error: {result['error']}"
            )
        else:
            print(
                f"{result['algorithm']:<15} {result['k']:<5} {result['clusters']:<10} {result['sequences']:<10} {result['status']:<15} -"
            )

    return {
        "total_attempts": total_diagrams,
        "successful_diagrams": successful_diagrams,
        "success_rate": successful_diagrams / total_diagrams
        if total_diagrams > 0
        else 0,
        "results_summary": results_summary,
        "output_directory": output_dir,
    }


def plot_sankey_diagram_adaptive(
    nodes,
    edges,
    node_to_cluster,
    colors,
    k,
    map_id,
    data_id,
    sampling_mode,
    algorithm,
    show_labels=True,
    summary=None,
    removed_counter=None,
    doublerow_annotation=False,
):
    """
    Adaptive Sankey diagram plotting function, supports arbitrary number of clusters and dynamic strategy nodes.
    :param nodes: Set of nodes
    :param edges: Set of edges, format: {(src, dst): value}
    :param node_to_cluster: Mapping from node to clusters, format: {node: [count1, count2, ..., countK]}
    :param colors: List of colors
    :param k: Number of clusters
    :param map_id: Map ID, used for file naming
    :param data_id: Data ID, used for file naming
    :param sampling_mode: Sampling mode
    :param algorithm: Algorithm name
    :param show_labels: Whether to show labels for each block, default is True
    :param summary: Cluster statistical information
    """

    # Node indexing
    node_indices = {node: i for i, node in enumerate(nodes)}
    source_indices = [node_indices[src] for src, dst in edges.keys()]
    target_indices = [node_indices[dst] for src, dst in edges.keys()]
    values = list(edges.values())

    # Node colors (based on clusters)
    node_colors = []
    cluster_colors = defaultdict(dict)

    # Dynamically generate mapping between strategy nodes and opening nodes
    def analyze_all_strategy_nodes():
        """Analyzes all nodes to generate mapping from strategy nodes to opening nodes."""
        strategy_to_nodes = {}
        strategy_labels = []
        opening_nodes = []

        for node in nodes:
            # If it's a strategy node (contains single quotes)
            if "'" in node and len(node) > 8:  # Strategy labels are usually long
                strategy_labels.append(node)
            # If it's an opening node (6-digit hex string)
            elif "【" not in node and len(node) <= 6:
                opening_nodes.append(node)

        # Assign colors to each strategy node
        strategy_colors = {}

        # Use a color library for different strategies
        base_strategy_colors = [
            "#5d009a", "#ef822f", "#FF6B6B", "#4ECDC4", "#45B7D1",
            "#96CEB4", "#FFEAA7", "#DDA0DD", "#98D8C8", "#F7DC6F",
            "#BB8FCE", "#85C1E2",
        ]

        for i, strategy in enumerate(strategy_labels):
            strategy_colors[strategy] = base_strategy_colors[
                i % len(base_strategy_colors)
            ]

        return strategy_labels, opening_nodes, strategy_colors

    # Generate a lighter version of a color based on the strategy node color
    def get_lighter_color(hex_color, factor=0.6):
        """Generates a lighter version of a given hex color.
        Args:
            hex_color (str): Hex color value, e.g., "#5d009a"
            factor (float): Lightness factor, between 0-1, smaller is lighter
        Returns:
            str: Lighter hex color value
        """
        color = hex_color.lstrip("#")
        r = int(color[0:2], 16)
        g = int(color[2:4], 16)
        b = int(color[4:6], 16)

        # Adjust towards white
        r = int(r + (255 - r) * factor)
        g = int(g + (255 - g) * factor)
        b = int(b + (255 - b) * factor)

        return "#{:02x}{:02x}{:02x}".format(r, g, b)

    # Dynamically generate strategy mappings
    strategy_labels, opening_nodes, strategy_colors = analyze_all_strategy_nodes()

    # Establish mapping from opening nodes to strategy nodes
    opening_to_strategy = {}
    for (source, target), value in edges.items():
        # If the edge goes from a strategy node to an opening node
        if "'" in source:
            opening_to_strategy[target] = source

    # Calculate color for each node
    incoming_nodes = {}
    for source, target in edges:
        if target not in incoming_nodes:
            incoming_nodes[target] = []
        incoming_nodes[target].append(source)

    for node in nodes:
        if "Cluster" in node:
            # Cluster labels use the corresponding cluster color
            cluster_index = int(node.split(" ")[1])
            node_colors.append(colors[cluster_index % len(colors)])
            cluster_colors[cluster_index] = colors[cluster_index % len(colors)]
        else:
            # All other nodes use cluster color or blended color
            cluster_counts = node_to_cluster[node]
            total_count = sum(cluster_counts)
            cluster_proportions = [count / total_count for count in cluster_counts]
            if max(cluster_proportions) == 1.0:
                dominant_cluster = cluster_proportions.index(max(cluster_proportions))
                node_colors.append(colors[dominant_cluster % len(colors)])
            else:
                mixed_color = [0, 0, 0]
                for i in range(k):
                    color = colors[i % len(colors)].lstrip("#")
                    r, g, b = (
                        int(color[0:2], 16),
                        int(color[2:4], 16),
                        int(color[4:6], 16),
                    )
                    mixed_color[0] += r * cluster_proportions[i]
                    mixed_color[1] += g * cluster_proportions[i]
                    mixed_color[2] += b * cluster_proportions[i]
                mixed_color = [int(c) for c in mixed_color]
                mixed_color_hex = "#{:02x}{:02x}{:02x}".format(*mixed_color)
                node_colors.append(mixed_color_hex)

    # Calculate node IDs in the downstream flow for each strategy node and opening node
    def calculate_strategy_flow_nodes():
        """
        Calculates node IDs for all downstream flow of each strategy node and opening node.
        Returns dictionary: {strategy_node: [list of all reachable node IDs]}
        """
        strategy_flow_nodes = {}

        # Build direct mapping from strategy nodes to opening nodes
        strategy_to_openings = {}
        for (source, target), value in edges.items():
            if (
                "'" in source
                and len(source) > 8
                and ("【" not in target and len(target) <= 6)
            ):
                if source not in strategy_to_openings:
                    strategy_to_openings[source] = []
                strategy_to_openings[source].append(target)

        # For each strategy node, find all reachable nodes
        for strategy_node in strategy_labels:
            reachable_nodes = set()

            if strategy_node in strategy_to_openings:
                op_nodes = strategy_to_openings[strategy_node]
                reachable_nodes.update(op_nodes)

                # Use BFS starting from each opening node to find all reachable downstream nodes
                for opening_node in op_nodes:
                    queue = [opening_node]
                    visited = set([opening_node])

                    while queue:
                        current_node = queue.pop(0)
                        for (source, target), value in edges.items():
                            if source == current_node and target not in visited:
                                visited.add(target)
                                queue.append(target)
                                reachable_nodes.add(target)

            strategy_flow_nodes[strategy_node] = [
                node_indices[node] for node in reachable_nodes if node in node_indices
            ]

        return strategy_flow_nodes

    # Calculate total inflow for each node (determines block height in Sankey)
    def calculate_node_flow_values():
        """
        Calculates total inflow for each node to determine block height.
        Returns dictionary: {node_id: total_flow}
        """
        node_flow_values = {}
        for node_id in range(len(nodes)):
            node_flow_values[node_id] = 0

        for (source, target), value in edges.items():
            if source in node_indices and target in node_indices:
                target_id = node_indices[target]
                node_flow_values[target_id] += value

        return node_flow_values

    # Get color and flow info for each strategy node
    def get_strategy_nodes_info():
        """
        Gets color and flow info for each strategy node and its downstream flow.
        Returns format: {
            strategy_node: {
                'node_colors': {node_id: color},
                'node_values': {node_id: flow_value},
                'total_flow': total_flow
            }
        }
        """
        strategy_to_all_flow_nodes = calculate_strategy_flow_nodes()
        node_flow_values = calculate_node_flow_values()
        strategy_info = {}

        for strategy_node, flow_node_ids in strategy_to_all_flow_nodes.items():
            strategy_info[strategy_node] = {
                "node_colors": {},
                "node_values": {},
                "total_flow": 0,
            }

            total_flow = 0
            for node_id in flow_node_ids:
                if node_id < len(node_colors):
                    strategy_info[strategy_node]["node_colors"][node_id] = node_colors[node_id]
                else:
                    strategy_info[strategy_node]["node_colors"][node_id] = "#808080"

                flow_val = node_flow_values.get(node_id, 0)
                strategy_info[strategy_node]["node_values"][node_id] = flow_val
                total_flow += flow_val

            strategy_info[strategy_node]["total_flow"] = total_flow

        return strategy_info

    def hex_to_rgb(hex_color):
        """Converts hex color string to RGB tuple."""
        if not isinstance(hex_color, str) or not hex_color.startswith("#"):
            return (128, 128, 128)
        try:
            hex_color = hex_color.lstrip("#")
            if len(hex_color) != 6:
                return (128, 128, 128)
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            return (r, g, b)
        except ValueError:
            return (128, 128, 128)

    def rgb_to_hex(r, g, b):
        """Converts RGB values to hex color string."""
        return "#{:02x}{:02x}{:02x}".format(
            max(0, min(255, int(r))), max(0, min(255, int(g))), max(0, min(255, int(b)))
        )

    # Calculate all successors of opening nodes
    def get_opening_successors():
        """
        Calculates all successors for each opening node.
        Returns: {opening_node: [list of successor node IDs]}
        """
        opening_successors = {}
        for node in nodes:
            if "【" not in node and len(node) <= 6 and node in node_indices:
                node_id = node_indices[node]
                successors = set()
                queue = [node]
                visited = {node}

                while queue:
                    current_node = queue.pop(0)
                    for (source, target), value in edges.items():
                        if source == current_node and target not in visited:
                            visited.add(target)
                            queue.append(target)
                            if target in node_indices:
                                successors.add(node_indices[target])

                opening_successors[node] = list(successors)
        return opening_successors

    def get_strategy_opening_first_successors_info():
        """
        Gets colors and flow info for the opening nodes and their first-level successors.
        Returns structured info for each strategy node.
        """
        node_flow_values = calculate_node_flow_values()
        strategy_to_openings = {}
        for (source, target), value in edges.items():
            if (
                "'" in source
                and len(source) > 8
                and ("【" not in target and len(target) <= 6)
            ):
                if source not in strategy_to_openings:
                    strategy_to_openings[source] = []
                strategy_to_openings[source].append(target)

        strategy_info = {}
        for strategy_node in strategy_labels:
            strategy_info[strategy_node] = {
                "opening_nodes": strategy_to_openings.get(strategy_node, []),
                "first_successors": {},
                "successor_colors": {},
                "successor_values": {},
                "total_successor_flow": {},
            }

            for opening_node in strategy_info[strategy_node]["opening_nodes"]:
                first_successors = []
                for (source, target), value in edges.items():
                    if source == opening_node and target in node_indices:
                        first_successors.append(node_indices[target])

                strategy_info[strategy_node]["first_successors"][opening_node] = first_successors
                suc_colors = {}
                suc_values = {}
                total_flow = 0

                for successor_id in first_successors:
                    suc_colors[successor_id] = node_colors[successor_id] if successor_id < len(node_colors) else "#808080"
                    f_val = node_flow_values.get(successor_id, 0)
                    suc_values[successor_id] = f_val
                    total_flow += f_val

                strategy_info[strategy_node]["successor_colors"][opening_node] = suc_colors
                strategy_info[strategy_node]["successor_values"][opening_node] = suc_values
                strategy_info[strategy_node]["total_successor_flow"][opening_node] = total_flow

        return strategy_info

    def calculate_strategy_mixed_colors(strategy_opening_info, nodes):
        """Calculates weighted blended colors for strategy nodes based on downstream flow."""
        strategy_mixed_colors = {}
        nodes_list = list(nodes)

        for strategy_node, info in strategy_opening_info.items():
            print(f"\nCalculating mixed color for strategy node {strategy_node}:")
            all_successors = {}
            total_flow = 0

            for opening_node in info["opening_nodes"]:
                if opening_node in info["successor_values"]:
                    s_values = info["successor_values"][opening_node]
                    s_colors = info["successor_colors"][opening_node]
                    print(f"  Opening Node {opening_node}: {len(s_values)} successors")
                    for s_id, f_val in s_values.items():
                        if f_val > 0:
                            all_successors[s_id] = {
                                "flow": f_val,
                                "color": s_colors.get(s_id, "#808080"),
                                "opening": opening_node,
                            }
                            total_flow += f_val

            if total_flow == 0 or not all_successors:
                strategy_mixed_colors[strategy_node] = "#808080"
                print(f"  No downstream flow, using default #808080")
                continue

            mixed_r, mixed_g, mixed_b = 0, 0, 0
            print(f"  Blending colors based on {len(all_successors)} successors:")
            for s_id, s_info in all_successors.items():
                f_val = s_info["flow"]
                n_color = s_info["color"]
                op_node = s_info["opening"]
                r, g, b = hex_to_rgb(n_color)
                weight = f_val / total_flow
                mixed_r += r * weight
                mixed_g += g * weight
                mixed_b += b * weight
                s_name = nodes_list[s_id] if s_id < len(nodes_list) else f"Node_{s_id}"
                print(f"    {s_name} (from {op_node}): Color {n_color}, Flow {f_val}, Weight {weight:.3f}")

            mixed_color = rgb_to_hex(mixed_r, mixed_g, mixed_b)
            strategy_mixed_colors[strategy_node] = mixed_color
            print(f"  Final mixed color: {mixed_color} | Total Flow: {total_flow}")

        return strategy_mixed_colors

    def validate_color(color_str):
        """Validates if string is a valid hex color format."""
        if not isinstance(color_str, str) or not color_str.startswith("#") or len(color_str) != 7:
            return False
        try:
            int(color_str[1:], 16)
            return True
        except ValueError:
            return False

    # Validate and fix color list
    validated_colors = []
    for i, color in enumerate(node_colors):
        if validate_color(color):
            validated_colors.append(color)
        else:
            print(f"Warning: Invalid color '{color}' at index {i}, changing to white")
            validated_colors.append("#FFFFFF")
    node_colors = validated_colors

    # Create node labels
    node_labels = []
    for node in nodes:
        if show_labels:
            node_labels.append(node)
        else:
            if "Cluster" in node:
                node_labels.append(node)
            else:
                node_labels.append("")

    # Plot Sankey
    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=4, thickness=20,
                    line=dict(color="black", width=1),
                    label=node_labels,
                    color=node_colors,
                ),
                textfont=dict(color="black", size=48),
                link=dict(
                    source=source_indices,
                    target=target_indices,
                    value=values,
                ),
                arrangement="snap",
                hoverinfo="none",
            )
        ]
    )

    annotations = []
    if doublerow_annotation:
        for i, (cluster_id, stats) in enumerate(summary.items()):
            color = cluster_colors[cluster_id % len(cluster_colors)]
            p1 = 5 - len(str(stats["count"]))
            p2 = 6 - len(str(f"{stats['max']:.2f}"))
            annotations.append(
                go.layout.Annotation(
                    text=f"<b>Cluster {cluster_id} | Size:{stats['count']} {' ' * p1} Avg:{stats['mean']:.2f}<br>"
                         f"          | Max:{stats['max']:.2f} {' ' * p2} Min:{stats['min']:.2f}</b>",
                    align="left", showarrow=False, xref="paper", yref="paper",
                    x=0.225, y=1.33 - 0.086 * (i + 1),
                    font=dict(size=30, color=color, family="Consolas, Monaco, monospace"),
                    bgcolor="rgba(255,255,255,0.8)", bordercolor=color, borderwidth=1, width=600,
                )
            )
    else:
        for i, (cluster_id, stats) in enumerate(summary.items()):
            color = cluster_colors[cluster_id % len(cluster_colors)]
            annotations.append(
                go.layout.Annotation(
                    text=f"<b>Cluster {cluster_id}</b> | Size: {stats['count']} Avg: {stats['mean']:.2f} Max: {stats['max']:.2f} Min: {stats['min']:.2f}",
                    align="left", showarrow=False, xref="paper", yref="paper",
                    x=0.00, y=1.28 - 0.036 * (i + 1),
                    font=dict(size=24, color=color),
                    bgcolor="rgba(255,255,255,0.8)", bordercolor=color, borderwidth=1, width=775,
                )
            )

    for cluster_id, count in removed_counter.items():
        color = cluster_colors[cluster_id % len(cluster_colors)]
        annotations.append(
            go.layout.Annotation(
                text=(f"Cluster {cluster_id}: Removed Reverse Edge Count={count}"),
                align="right", showarrow=False, xref="paper", yref="paper",
                x=0.25, y=1.06 - 0.017 * (cluster_id + 1),
                font=dict(size=10, color=color),
                bgcolor="rgba(255,255,255,0.8)", bordercolor=color, borderwidth=1, width=240,
            )
        )

    fig.update_layout(annotations=annotations, margin=dict(l=10, r=10, t=240, b=10))

    # Path generation logic
    if doublerow_annotation:
        dir_png = get_output_dir(OutputPaths.SANKEY_D, map_id, "png")
        output_filename_png = f"{dir_png}/{algorithm}_sankey_{map_id}_{data_id}_{k}_{sampling_mode}_d.png"
    else:
        dir_png = get_output_dir(OutputPaths.SANKEY, map_id, "png")
        output_filename_png = f"{dir_png}/{algorithm}_sankey_{map_id}_{data_id}_{k}_{sampling_mode}.png"

    fig.write_image(output_filename_png, width=800, height=1200, scale=1)
    print(f"Sankey diagram saved as: {output_filename_png}")


def visualize_specific_pattern_comprehensive_tactic(
    all_cluster_results,
    state_distance_matrix,
    log_distance_matrix,
    log_object,
    game_results,
    state_log,
    seed=42,
    enable_sampling=True,
    sampling_ratio=0.5,
    sampling_mode="completely_random",
    width=300,
    height=450,
    map_id="map1",
    data_id="data1",
    show_labels=True,
    enable_plot=True,
    doublerow_annotation=False,
):
    """
    Function to draw Sankey diagrams based on existing clustering results.
    :param all_cluster_results: List containing all clustering results; each element is a dict with algorithm, k, labels, etc.
    :param state_distance_matrix: State distance matrix
    :param log_distance_matrix: Log distance matrix
    :param log_object: List of fitness values
    :param game_results: List of game results
    :param state_log: Sequence of state logs
    :param enable_sampling: Whether to sample from each cluster
    :param sampling_ratio: Sampling ratio (0.0-1.0)
    :param sampling_mode: Sampling mode:
                         - "completely_random": Fully random sampling
                         - "top_random": Sort by score descending, take top ratio
                         - "back_random": Sort by score ascending, take bottom ratio
    :param width: Image width
    :param height: Image height
    :param map_id: Map ID
    :param data_id: Data ID
    :param show_labels: Whether to display labels
    :return: Processing result summary statistics
    """

    print("Starting pattern analysis based on existing clustering results...")

    # Create output directory
    output_dir = get_output_dir(OutputPaths.SANKEY_TACTIC)
    print(f"Created directory: {output_dir}")

    action_dict = create_action_dictionary()

    def classify_start_node(start_node):
        strategy_labels = analyze_strategy_node(start_node, action_dict)
        return strategy_labels

    def classify_part_node(part_node):
        clean = re.sub(r"[【】]", "", part_node)
        strategy_labels = analyze_strategy_node(clean, action_dict)
        return strategy_labels

    # Get all sequence data
    log_path = action_log_path
    result_path = game_result_path
    all_sequences = get_marked_sequences(log_path, result_path, min_support=0.01)

    # Statistics info
    total_diagrams = 0
    successful_diagrams = 0
    results_summary = []

    # Create global statistics dictionary
    global_node_statistics = {}

    # Iterate through all clustering results
    for algorithm, cluster_labels in all_cluster_results.items():
        # 1. Determine optimal cluster count and labels
        k = int(algorithm.split("_")[-1].replace("k", ""))
        labels = cluster_labels

        print(f"\n--- Processing {algorithm.upper()} with k={k} ---")

        # 2. Generate dynamic color scheme
        dynamic_colors = generate_dynamic_colors(k)

        if enable_sampling:
            # Multi-mode sampling logic
            print(
                f"  Sampling {sampling_ratio * 100:.0f}% of sequences from each cluster (mode: {sampling_mode})..."
            )
            sampled_indices = []

            np.random.seed(42)

            # Get indices and corresponding scores for each cluster
            for cluster_id in range(k):
                cluster_indices = [
                    i for i, label in enumerate(labels) if label == cluster_id
                ]
                if cluster_indices:
                    # Get scores for these indices
                    cluster_scores = [log_object[i] for i in cluster_indices]
                    cluster_data = list(zip(cluster_indices, cluster_scores))

                    # Calculate number of samples
                    n_sample = max(1, int(len(cluster_indices) * sampling_ratio))

                    # Sample based on mode
                    if sampling_mode == "completely_random":
                        sampled_cluster_indices = np.random.choice(
                            cluster_indices, n_sample, replace=False
                        )
                        sampled_indices.extend(sampled_cluster_indices)

                    elif sampling_mode == "top_random":
                        # Sort by score descending, take top n_sample
                        cluster_data.sort(key=lambda x: x[1], reverse=True)
                        sampled_cluster_indices = [
                            data[0] for data in cluster_data[:n_sample]
                        ]
                        sampled_indices.extend(sampled_cluster_indices)

                    elif sampling_mode == "back_random":
                        # Sort by score ascending, take first n_sample (bottom scores)
                        cluster_data.sort(key=lambda x: x[1])
                        sampled_cluster_indices = [
                            data[0] for data in cluster_data[:n_sample]
                        ]
                        sampled_indices.extend(sampled_cluster_indices)

                    else:
                        print(
                            f"  Warning: Unknown sampling mode '{sampling_mode}', using completely_random"
                        )
                        sampled_cluster_indices = np.random.choice(
                            cluster_indices, n_sample, replace=False
                        )
                        sampled_indices.extend(sampled_cluster_indices)

            # Build new sequence list based on sampled indices
            sampled_sequences = [
                (all_sequences[i][1], all_sequences[i][2]) for i in sampled_indices
            ]
            sampled_labels = [labels[i] for i in sampled_indices]
            sampled_results = [all_sequences[i][2] for i in sampled_indices]

            # Group sequences by cluster using sampled data
            cluster_sequences = get_sequences_by_cluster(
                sampled_labels, sampled_sequences
            )
            print(
                f"  Sampled {len(sampled_sequences)} sequences from {len(labels)} total sequences"
            )
        else:
            # Use all data
            sequences = [
                (all_sequences[i][1], all_sequences[i][2]) for i in range(len(labels))
            ]
            cluster_sequences = get_sequences_by_cluster(labels, sequences)
            print(f"  Using all {len(sequences)} sequences")
            sampled_labels = [labels[i] for i in sequences]
            sampled_results = [all_sequences[i][2] for i in sampled_labels]

        info = defaultdict(lambda: {"count": 0, "values": []})

        for label, value in zip(sampled_labels, sampled_results):
            info[label]["count"] += 1
            info[label]["values"].append(value)

        # Calculate statistics
        summary = {}
        for label, data in info.items():
            vals = data["values"]
            summary[label] = {
                "count": data["count"],
                "max": max(vals),
                "min": min(vals),
                "mean": round(sum(vals) / len(vals), 2),
            }

        # Extract nodes and edges
        nodes = set()
        edges = defaultdict(int)
        node_to_cluster = defaultdict(lambda: [0] * k)

        def extract_nodes(sequence: str):
            """
            Returns:
                nodes -> list of original 【...】 parts
                first -> inner text of the first 【】 (first 6 chars)
            """
            nodes = []
            first = None
            start = 0
            while True:
                start = sequence.find("【", start)
                if start == -1:
                    break
                end = sequence.find("】", start)
                if end == -1:
                    break
                nodes.append(sequence[start: end + 1])
                if first is None:
                    inner = sequence[start + 1: end]
                    first = inner[:6]
                start = end + 1
            return nodes, first

        # Score normalization logic
        print("Collecting scores for normalization...")
        all_scores = []
        for cluster, samples in cluster_sequences.items():
            for sequence, score in samples:
                all_scores.append(score)

        if all_scores:
            min_score = min(all_scores)
            max_score = max(all_scores)
            print(f"Score range: [{min_score:.3f}, {max_score:.3f}]")

            if max_score == min_score:
                print("All scores are equal, using uniform weights")

                def normalize_score(score):
                    return 1.0
            else:
                def normalize_score(score):
                    return (score - min_score) / (max_score - min_score)
        else:
            print("No scores found, using uniform weights")

            def normalize_score(score):
                return 1.0

        def would_create_cycle(source, target, edges_dict):
            """Checks if adding an edge from source to target creates a cycle."""
            visited = set()
            stack = [target]
            while stack:
                current = stack.pop()
                if current == source:
                    return True
                if current in visited:
                    continue
                visited.add(current)
                for src, dst in edges_dict.keys():
                    if src == current and dst not in visited:
                        stack.append(dst)
            return False

        def process_tactics_lists(indexed_tactics_lists, node_to_cluster, color_mode="fresh"):
            edges = defaultdict(int)
            all_renamed_nodes = []
            temp_nodes = []
            temp_edges = defaultdict(int)
            with_suffix_counter = Counter()
            without_suffix_counter = Counter()

            for sublist, cluster_id in indexed_tactics_lists:
                local_counter = 1
                renamed_sequence = []
                for tactic in sublist:
                    renamed = f"{tactic}({local_counter})"
                    local_counter += 1
                    renamed_sequence.append(renamed)
                    all_renamed_nodes.append(renamed)
                    with_suffix_counter[renamed] += 1
                    without_suffix_counter[tactic] += 1
                    if color_mode == "fresh":
                        node_to_cluster[renamed][cluster_id] += 1
                    else:
                        node_to_cluster[tactic][cluster_id] += 1

                temp_nodes.append(f"Cluster {cluster_id}")
                end = None
                for i in range(len(renamed_sequence) - 1):
                    src = renamed_sequence[i]
                    dst = renamed_sequence[i + 1]
                    edges[(src, dst)] += 1
                    end = dst
                if end is None:
                    end = renamed_sequence[-1]
                temp_edges[(end, f"Cluster {cluster_id}")] += 1

            all_renamed_nodes.extend(temp_nodes)
            edges.update(temp_edges)
            seen = set()
            unique_nodes = []
            for node in all_renamed_nodes:
                if node not in seen:
                    seen.add(node)
                    unique_nodes.append(node)

            total_nodes = sum(with_suffix_counter.values())
            with_suffix_stats = {n: {"frequency": c, "proportion": c / total_nodes} for n, c in
                                 with_suffix_counter.items()}
            without_suffix_stats = {n: {"frequency": c, "proportion": c / total_nodes} for n, c in
                                    without_suffix_counter.items()}

            return set(unique_nodes), dict(edges), with_suffix_stats, without_suffix_stats

        def save_node_statistics(with_suffix_stats, without_suffix_stats, tactic_dict, algorithm, map_id, data_id, k,
                                 sampling_mode, global_stats_dict, tactic_length_dict):
            """Saves node statistics to text files."""
            reverse_tactic_dict = {v: k for k, v in tactic_dict.items()}
            key = f"{algorithm}_{k}_{sampling_mode}"

            def replace_node_name(node_name):
                if node_name.startswith("Cluster "):
                    return node_name
                match = re.match(r"^(.+)\((\d+)\)$", str(node_name))
                if match:
                    base_node, suffix = match.group(1), match.group(2)
                    return f"{reverse_tactic_dict.get(base_node, base_node)}({suffix})"
                return reverse_tactic_dict.get(node_name, node_name)

            def encode_real_name(real_name):
                tactic_code_dict = {
                    "greedy": 1, "threat-focused": 2, "exfiltration": 3, "massing": 4, "feint": 5,
                    "sacrificial feint": 6, "inefficiency": 7, "independently": 8, "distributionally": 9,
                    "centrally": 10, "sustained coordination": 11, "switched coordination": 12,
                    "sustained attack": 13, "sustained defense": 14, "sustained hybrid": 15,
                    "switched tactic": 16, "inefficient tactic": 17, "unknown tactic": 18,
                }
                code = np.zeros(len(tactic_code_dict), dtype=np.uint8)
                keys_found = re.findall(r"'([^']*)'", real_name)
                for k in keys_found:
                    if k in tactic_code_dict:
                        code[tactic_code_dict[k] - 1] = 1
                return "".join(map(str, code))

            def merge_stats_by_real_name(stats_dict):
                merged_stats = {}
                for node, stats in stats_dict.items():
                    real_name = replace_node_name(node)
                    match = re.match(r"^(.+)\((\d+)\)$", node)
                    node_length = tactic_length_dict[match.group(1)] if match else tactic_length_dict[node]
                    code = encode_real_name(real_name)
                    if code not in merged_stats:
                        merged_stats[code] = {"frequency": 0, "proportion": 0.0, "length": []}
                    merged_stats[code]["frequency"] += stats["frequency"]
                    merged_stats[code]["length"].append(node_length)

                total_freq = sum(s["frequency"] for s in merged_stats.values())
                for c in merged_stats:
                    merged_stats[c]["proportion"] = merged_stats[c]["frequency"] / total_freq if total_freq > 0 else 0
                return merged_stats

            merged_with = merge_stats_by_real_name(with_suffix_stats)
            merged_without = merge_stats_by_real_name(without_suffix_stats)

            if key not in global_stats_dict:
                global_stats_dict[key] = {
                    "algorithm": algorithm, "k": k, "sampling_mode": sampling_mode,
                    "map_id": map_id, "data_id": data_id,
                    "with_suffix_stats": [], "without_suffix_stats": [],
                }
            global_stats_dict[key]["with_suffix_stats"].append(merged_with)
            global_stats_dict[key]["without_suffix_stats"].append(merged_without)

        # Extract behavioral patterns
        print("Extracting behavioral patterns...")
        total_sequences = 0
        all_tactics = set()
        raw_tactic_lists = []
        raw_tactic_length_dict = defaultdict(Counter)

        for cluster, samples in cluster_sequences.items():
            for sequence, score in samples:
                parts, first_node = extract_nodes(sequence)
                if first_node and len(parts) != 0:
                    parts_tactic = [classify_part_node(part) for part in parts]
                    raw_tactic_lists.append((parts_tactic, cluster))
                    all_tactics.update(parts_tactic)
                    for part in parts:
                        p_tactic = classify_part_node(part)
                        raw_tactic_length_dict[p_tactic][len(part) - 2] += 1

        unique_tactics = list(OrderedDict.fromkeys(all_tactics))
        tactic_dict = {t: f"T{i + 1}" for i, t in enumerate(unique_tactics)}
        tactic_length_dict = {
            tid: sum(ln * count for ln, count in raw_tactic_length_dict[t].items()) / sum(
                raw_tactic_length_dict[t].values())
            for t, tid in tactic_dict.items()
        }

        indexed_tactics_lists = [
            ([tactic_dict[t] for t in t_list], cluster)
            for t_list, cluster in raw_tactic_lists
        ]

        color_mode = "unite"
        nodes, edges, with_stats, without_stats = process_tactics_lists(indexed_tactics_lists, node_to_cluster,
                                                                        color_mode)

        save_node_statistics(with_stats, without_stats, tactic_dict, algorithm, map_id, data_id, k, sampling_mode,
                             global_node_statistics, tactic_length_dict)

        removed_counter = Counter()
        print("Generating Sankey diagram...")
        if enable_plot:
            plot_sankey_diagram_adaptive_tactic(
                nodes, edges, node_to_cluster, dynamic_colors, k, map_id, data_id, sampling_mode, algorithm,
                show_labels=show_labels, summary=summary, removed_counter=removed_counter, color_mode=color_mode,
                doublerow_annotation=doublerow_annotation
            )
            print(f"Comprehensive pattern analysis completed with {k} clusters")
        total_diagrams += 1

    def batch_process_global_stats(global_stats_dict):
        """Processes global stats to calculate mean and standard deviation."""
        output_dir = f"output_sankey_tactic/{map_id}/stats_summary"
        os.makedirs(output_dir, exist_ok=True)

        all_codes_without_suffix = set()
        for key, data in global_stats_dict.items():
            for stats in data["without_suffix_stats"]:
                all_codes_without_suffix.update(stats.keys())

        node_freq_data = {code: [] for code in all_codes_without_suffix}
        node_prop_data = {code: [] for code in all_codes_without_suffix}

        for key, data in global_stats_dict.items():
            for stats in data["without_suffix_stats"]:
                for code in all_codes_without_suffix:
                    s = stats.get(code, {"frequency": 0, "proportion": 0.0})
                    node_freq_data[code].append(s["frequency"])
                    node_prop_data[code].append(s["proportion"])

        # Statistics summary logic...
        return output_dir

    def integrate_all_configurations_stats(global_stats_dict):
        """Integrates statistics across all algorithm configurations."""
        print("Integrating statistics across all algorithm configurations...")
        output_dir = f"output_sankey_tactic/{map_id}/stats_summary"
        os.makedirs(output_dir, exist_ok=True)

        all_codes = set()
        for key, data in global_stats_dict.items():
            for stats in data["with_suffix_stats"]:
                all_codes.update(stats.keys())

        cross_config_freq = {c: [] for c in all_codes}
        cross_config_prop = {c: [] for c in all_codes}
        cross_config_len = {c: [] for c in all_codes}

        for key, data in global_stats_dict.items():
            for stats in data["without_suffix_stats"]:
                for code in all_codes:
                    if code in stats:
                        cross_config_freq[code].append(stats[code]["frequency"])
                        cross_config_prop[code].append(stats[code]["proportion"])
                        cross_config_len[code].extend(stats[code]["length"])
                    else:
                        cross_config_freq[code].append(0)
                        cross_config_prop[code].append(0.0)

        sorted_codes = sorted(all_codes, key=lambda x: int(x, 2))
        output_file = f"{output_dir}/node_statistics_summary_without_suffix_{map_id}_{data_id}_{sampling_mode}.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("Binary Code\t\tMean Freq\tStd Freq\tMean Prop\tStd Prop\tMean Length\n")
            for code in sorted_codes:
                f.write(f"{code}\t\t{np.mean(cross_config_freq[code]):.6f}\t{np.std(cross_config_freq[code]):.6f}\t"
                        f"{np.mean(cross_config_prop[code]):.6f}\t{np.std(cross_config_prop[code]):.6f}\t"
                        f"{np.mean(cross_config_len[code]) if cross_config_len[code] else 0:.6f}\n")
        return output_file

    batch_process_global_stats(global_node_statistics)
    integrate_all_configurations_stats(global_node_statistics)

    print(f"\n{'=' * 60}\nSANKEY DIAGRAM GENERATION SUMMARY\n{'=' * 60}")
    print(f"Total processing attempts: {total_diagrams}")
    return {"total_attempts": total_diagrams, "output_directory": output_dir}


def plot_sankey_diagram_adaptive_tactic(
    nodes,
    edges,
    node_to_cluster,
    colors,
    k,
    map_id,
    data_id,
    sampling_mode,
    algorithm,
    show_labels=True,
    summary=None,
    removed_counter=None,
    color_mode="fresh",
    doublerow_annotation=False,
):
    """
    Adaptive Sankey diagram plotting function, supporting any number of clusters and dynamic tactic nodes.
    :param nodes: Set of nodes
    :param edges: Set of edges, format: {(src, dst): value}
    :param node_to_cluster: Mapping from node to clusters, format: {node: [count1, count2, ..., countK]}
    :param colors: List of colors
    :param k: Number of clusters
    :param map_id: Map ID, used for file naming
    :param data_id: Data ID, used for file naming
    :param sampling_mode: Sampling mode
    :param algorithm: Algorithm name
    :param show_labels: Whether to display labels for each block, defaults to True
    :param summary: Cluster statistics information
    """

    # Node indexing
    node_indices = {node: i for i, node in enumerate(nodes)}
    source_indices = [node_indices[src] for src, dst in edges.keys()]
    target_indices = [node_indices[dst] for src, dst in edges.keys()]
    values = list(edges.values())

    # Node colors (based on clusters)
    node_colors = []
    cluster_colors = defaultdict(dict)

    # Dynamically generate mapping between tactic nodes and opening nodes
    def analyze_all_strategy_nodes():
        """Analyzes all nodes to generate mapping from tactic nodes to opening nodes."""
        strategy_to_nodes = {}
        strategy_labels = []
        opening_nodes = []

        for node in nodes:
            # If it's a tactic node (contains single quote labels)
            if "'" in node and len(node) > 8:  # Tactic labels are usually longer
                strategy_labels.append(node)
            # If it's an opening node (6-digit hex identifier)
            elif "【" not in node and len(node) <= 6:
                opening_nodes.append(node)

        # Assign colors to each tactic node
        strategy_colors = {}

        # Use color library to generate colors for different tactics
        base_strategy_colors = [
            "#5d009a", "#ef822f", "#FF6B6B", "#4ECDC4", "#45B7D1",
            "#96CEB4", "#FFEAA7", "#DDA0DD", "#98D8C8", "#F7DC6F",
            "#BB8FCE", "#85C1E2",
        ]

        for i, strategy in enumerate(strategy_labels):
            strategy_colors[strategy] = base_strategy_colors[
                i % len(base_strategy_colors)
                ]

        return strategy_labels, opening_nodes, strategy_colors

    # Generate lighter version based on tactic node color
    def get_lighter_color(hex_color, factor=0.6):
        """Generates a lighter version of a given hex color.
        Args:
            hex_color (str): Hex color value, e.g., "#5d009a"
            factor (float): Lightness factor, 0-1, smaller is lighter.
        Returns:
            str: Lighter hex color value
        """
        color = hex_color.lstrip("#")
        r = int(color[0:2], 16)
        g = int(color[2:4], 16)
        b = int(color[4:6], 16)

        # Adjust towards white
        r = int(r + (255 - r) * factor)
        g = int(g + (255 - g) * factor)
        b = int(b + (255 - b) * factor)

        return "#{:02x}{:02x}{:02x}".format(r, g, b)

    # Dynamically generate strategy mappings
    strategy_labels, opening_nodes, strategy_colors = analyze_all_strategy_nodes()

    # Establish mapping from opening nodes to strategy nodes
    opening_to_strategy = {}
    for (source, target), value in edges.items():
        # If edge goes from a strategy node to an opening node
        if (
                "'" in source
                and len(source) > 8
                and ("【" not in target and len(target) <= 6)
        ):
            opening_to_strategy[target] = source

    # Calculate color for each node
    incoming_nodes = {}
    for source, target in edges:
        if target not in incoming_nodes:
            incoming_nodes[target] = []
        incoming_nodes[target].append(source)

    for node in nodes:
        if "Cluster" in node:
            # Cluster labels use their respective cluster colors
            cluster_index = int(node.split(" ")[1])
            node_colors.append(colors[cluster_index % len(colors)])
            cluster_colors[cluster_index] = colors[cluster_index % len(colors)]
        else:
            # Other nodes use cluster colors or mixed colors
            cluster_counts = None
            if color_mode == "fresh":
                cluster_counts = node_to_cluster[node]
            elif color_mode == "unite":
                unite_node = re.sub(r"\(.*?\)", "", node)
                cluster_counts = node_to_cluster[unite_node]

            total_count = sum(cluster_counts)
            if total_count == 0:
                node_colors.append("black")
            else:
                cluster_proportions = [count / total_count for count in cluster_counts]
                if max(cluster_proportions) == 1.0:
                    dominant_cluster = cluster_proportions.index(max(cluster_proportions))
                    node_colors.append(colors[dominant_cluster % len(colors)])
                else:
                    mixed_color = [0, 0, 0]
                    for i in range(k):
                        color = colors[i % len(colors)].lstrip("#")
                        r, g, b = (
                            int(color[0:2], 16),
                            int(color[2:4], 16),
                            int(color[4:6], 16),
                        )
                        mixed_color[0] += r * cluster_proportions[i]
                        mixed_color[1] += g * cluster_proportions[i]
                        mixed_color[2] += b * cluster_proportions[i]
                    mixed_color = [int(c) for c in mixed_color]
                    mixed_color_hex = "#{:02x}{:02x}{:02x}".format(*mixed_color)
                    node_colors.append(mixed_color_hex)

    # Validate that all colors in node_colors are valid; otherwise set to white
    def validate_color(color_str):
        """Validates if a color string is a valid hex color format (#RRGGBB)."""
        if not isinstance(color_str, str):
            return False
        if not color_str.startswith("#") or len(color_str) != 7:
            return False
        hex_part = color_str[1:]
        try:
            int(hex_part, 16)
            return True
        except ValueError:
            return False

    # Validate and correct the color list
    white_color = "#FFFFFF"
    validated_colors = []
    for i, color in enumerate(node_colors):
        if validate_color(color):
            validated_colors.append(color)
        else:
            print(f"Warning: Invalid color '{color}' at index {i}, changing to white")
            validated_colors.append(white_color)
    node_colors = validated_colors

    # Create node labels based on show_labels parameter
    node_labels = []
    for node in nodes:
        if show_labels:
            # Show all node labels
            node_labels.append(node)
        else:
            # Only show labels for cluster nodes
            if "Cluster" in node:
                node_labels.append(node)
            else:
                node_labels.append("")

    # Plot Sankey Diagram
    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=4,
                    thickness=20,
                    line=dict(color="black", width=1),
                    label=node_labels,
                    color=node_colors,
                ),
                textfont=dict(color="black", size=48),
                link=dict(
                    source=source_indices,
                    target=target_indices,
                    value=values,
                ),
                arrangement="snap",
                hoverinfo="none",
            )
        ]
    )

    # Add annotations
    annotations = []
    if doublerow_annotation:
        # Add a colored line of text for each cluster (double-row format)
        for i, (cluster_id, stats) in enumerate(summary.items()):
            color = cluster_colors[cluster_id % len(cluster_colors)]
            placeholder_1 = 5 - len(str(stats["count"]))
            placeholder_2 = 6 - len(str(f"{stats['max']:.2f}"))

            annotations.append(
                go.layout.Annotation(
                    text=f"<b>Cluster {cluster_id} | Size:{stats['count']} {' ' * placeholder_1} Avg:{stats['mean']:.2f}<br>"
                         f"          | Max:{stats['max']:.2f} {' ' * placeholder_2} Min:{stats['min']:.2f}</b>",
                    align="left",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.225,
                    y=1.33 - 0.086 * (i + 1),
                    font=dict(
                        size=30, color=color, family="Consolas, Monaco, monospace"
                    ),
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor=color,
                    borderwidth=1,
                    width=600,
                )
            )
    else:
        # Standard single-row annotation for each cluster
        for i, (cluster_id, stats) in enumerate(summary.items()):
            color = cluster_colors[cluster_id % len(cluster_colors)]
            annotations.append(
                go.layout.Annotation(
                    text=f"<b>Cluster {cluster_id}</b> | Size: {stats['count']} Avg: {stats['mean']:.2f} Max: {stats['max']:.2f} Min: {stats['min']:.2f}",
                    align="left",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.00,
                    y=1.28 - 0.036 * (i + 1),
                    font=dict(size=24, color=color),
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor=color,
                    borderwidth=1,
                    width=775,
                )
            )

    for cluster_id, count in removed_counter.items():
        color = cluster_colors[cluster_id % len(cluster_colors)]
        annotations.append(
            go.layout.Annotation(
                text=(f"Cluster {cluster_id}: Removed Reverse Edge Count={count}"),
                align="right",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.25,
                y=1.06 - 0.017 * (cluster_id + 1),
                font=dict(size=10, color=color),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor=color,
                borderwidth=1,
                width=240,
            )
        )

    fig.update_layout(annotations=annotations, margin=dict(l=10, r=10, t=240, b=10))

    # Define directory based on doublerow_annotation flag
    base_dir = "output_sankey_tactic_d" if doublerow_annotation else "output_sankey_tactic"
    suffix = "_d" if doublerow_annotation else ""

    output_dir_pdf = f"{base_dir}/{map_id}/pdf"
    output_dir_png = f"{base_dir}/{map_id}/png"

    for directory in [output_dir_pdf, output_dir_png]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    # Generate file names based on map_id, data_id, k, and sampling_mode
    output_filename_pdf = f"{output_dir_pdf}/{algorithm}_sankey_tactic_{map_id}_{data_id}_{k}_{sampling_mode}{suffix}.pdf"
    output_filename_png = f"{output_dir_png}/{algorithm}_sankey_tactic_{map_id}_{data_id}_{k}_{sampling_mode}{suffix}.png"

    # Save PNG image
    fig.write_image(output_filename_png, width=800, height=1200, scale=1)
    print(f"Sankey diagram saved as: {output_filename_pdf}, {output_filename_png}")


# Example Usage
if __name__ == "__main__":
    # Load Primary BKTree
    primary_bk_tree = load_bk_tree_from_file(primary_bktree_path)
    secondary_bk_trees = {}

    print("Root cluster ID:", primary_bk_tree.root.cluster_id)
    print("Root state:", primary_bk_tree.root.state)
    print("Root children:", primary_bk_tree.root.children.keys())
    cluster_count = get_max_cluster_id(primary_bk_tree)
    print("Cluster count:", cluster_count)

    # Load Secondary BKTrees for each cluster
    for cluster_id in range(1, cluster_count + 1):
        secondary_bktree_path = f"{secondary_bktree_prefix}_{cluster_id}.json"
        secondary_bk_trees[cluster_id] = load_bk_tree_from_file(secondary_bktree_path)
        print(f"Secondary BKTree for cluster {cluster_id}:")
        print("Root cluster ID:", secondary_bk_trees[cluster_id].root.cluster_id)
        print("Root state:", secondary_bk_trees[cluster_id].root.state)
        print("Root children:", secondary_bk_trees[cluster_id].root.children.keys())

    print("########################################################################")
    state_node_dict, reverse_dict = read_state_node_file(state_node_path)
    state_log = read_node_log_file(node_log_path)

    # Calculate and save state distance matrix
    state_distance_matrix = calculate_and_save_distance_matrix(
        reverse_dict, custom_distance, secondary_bk_trees, distance_matrix_folder
    )

    # Calculate and save DTW (Dynamic Time Warping) distance matrix for logs
    log_distance_matrix = calculate_and_save_dtw_distance_matrix(
        state_log, state_distance_matrix, distance_matrix_folder
    )

    game_results = read_game_result_file(game_result_path)
    # log_object represents fitness values (sum of specific log indices)
    log_object = [log[2] + log[3] for log in game_results]

    grid_x, grid_y, grid_z = get_fitness_landscape(
        log_distance_matrix, log_object, n=None
    )

    print("########################################################################")

    # Plot fitness-based state transition diagrams and fitness landscapes
    # using multiple clustering algorithms and different cluster counts (k)
    all_cluster_results = visualize_top_k_data_state_fitness_landscape(
        state_distance_matrix,
        log_distance_matrix,
        log_object,
        game_results,
        state_log,
        grid_x,
        grid_y,
        grid_z,
        topk=1,
        k_values=[3],  # Test different cluster counts (e.g., [3, 4, 5, 6])
        algorithms=["agglomerative"],  # Options: 'kmeans', 'gmm', 'agglomerative', 'birch'
        width=300,
        height=450,
        save_fig=True,  # Enable saving figures
        enable_heatmap=True  # Enable heatmap visualization
    )

    # Batch generate Sankey diagrams based on the obtained clustering results
    print("Generating comprehensive Sankey diagrams based on clustering results...")
    print(f"Processing {len(all_cluster_results)} clustering results...")

    # Generate standard comprehensive Sankey diagrams
    sankey_results = visualize_specific_pattern_comprehensive(
        all_cluster_results,  # List of clustering results
        state_distance_matrix,  # State distance matrix
        log_distance_matrix,  # Log distance matrix
        log_object,  # Fitness values list
        game_results,  # Game results list
        state_log,  # State log sequences
        seed=42,  # Random seed
        enable_sampling=True,  # Enable sampling
        sampling_ratio=0.01,  # Sampling ratio (1%)
        sampling_mode="back_random",  # Modes: completely_random, top_random, back_random
        width=300,
        height=450,
        map_id=map_id,  # Use global map_id variable
        data_id=data_id,  # Use global data_id variable
        show_labels=False  # Label visibility
    )

    # Generate tactic-specific comprehensive Sankey diagrams
    sankey_results = visualize_specific_pattern_comprehensive_tactic(
        all_cluster_results,  # List of clustering results
        state_distance_matrix,  # State distance matrix
        log_distance_matrix,  # Log distance matrix
        log_object,  # Fitness values list
        game_results,  # Game results list
        state_log,  # State log sequences
        seed=42,  # Random seed
        enable_sampling=True,  # Enable sampling
        sampling_ratio=0.01,  # Sampling ratio (0.01)
        sampling_mode="back_random",  # Modes: completely_random, top_random, back_random
        width=300,
        height=450,
        map_id=map_id,  # Global map_id
        data_id=data_id,  # Global data_id
        show_labels=False,  # Label visibility
        enable_plot=True  # Enable plotting
    )

    print(f"Sankey diagram generation completed:")
    print(f"  Total attempts: {sankey_results['total_attempts']}")
    print(f"  Successful diagrams: {sankey_results['successful_diagrams']}")
    print(f"  Success rate: {sankey_results['success_rate'] * 100:.1f}%")
    print(f"  Output directory: {sankey_results['output_directory']}")

    # Additional Analysis Options (Commented Out):
    # - visualize_top_k_data_pattern: Analyze top K data patterns
    # - visualize_top_k_data_pattern_comprehensive: Search for optimal k within a range
