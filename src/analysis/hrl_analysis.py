"""
HRL-IMCBS Data Analysis Module

Analyze and visualize the state clustering results of the HRL algorithm
"""

import json
import math
import os
import time
from sklearn.manifold import MDS
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.pyplot as plt
from src.distance.base import CustomDistance
from scipy.interpolate import griddata
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans

# Import configuration
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    get_data_paths,
    get_output_dir,
    OutputPaths,
    get_cache_path,
    DEFAULT_MAP_ID,
    DEFAULT_DATA_ID,
)

custom_distance = CustomDistance(threshold=0.5)

# Dataset configuration - switch datasets by modifying here
# Available datasets: MarineMicro_MvsM_4 (data_ids: 6), MarineMicro_MvsM_4_dist (data_id: 1), MarineMicro_MvsM_8 (data_id: 1)
map_id = DEFAULT_MAP_ID  # Or specify directly like "MarineMicro_MvsM_8m"
data_id = DEFAULT_DATA_ID  # Or specify directly like "1"
# map_id = "MarineMicro_MvsM_4_dist"
# data_id = "1"
# map_id = "MarineMicro_MvsM_8"
# data_id = "1"

# Use config module to get data paths
paths = get_data_paths(map_id, data_id)
distance_matrix_folder = paths["distance_matrix_folder"]
primary_bktree_path = paths["primary_bktree_path"]
secondary_bktree_prefix = paths["secondary_bktree_prefix"]
state_node_path = paths["state_node_path"]
node_log_path = paths["node_log_path"]
game_result_path = paths["game_result_path"]
action_log_path = paths["action_log_path"]

# Define global color list
COLOR_LIST = [
    "rgb(255, 99, 132)",  # Red
    "rgb(54, 162, 235)",  # Blue
    "rgb(255, 159, 64)",  # Orange
    "rgb(75, 192, 192)",  # Cyan
    "rgb(153, 102, 255)",  # Purple
    "rgb(255, 0, 255)",  # Pink
    "rgb(0, 255, 0)",  # Green
    "rgb(0, 255, 255)",  # Teal
    "rgb(255, 0, 0)",  # Pure Red
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
        Recursively find the BKTreeNode with the specified cluster_id
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

    :param file_path: File path
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
    max_cluster_id = [0]  # Use list to store max value for modification in recursion
    if bk_tree.root:
        find_max_cluster_id(bk_tree.root, max_cluster_id)
    return max_cluster_id[0]


def read_state_node_file(file_path):
    """
    Read file and store as dictionary
    :param file_path: File path
    :return: Dictionary format data
    """
    state_node_dict = {}
    with open(file_path, "r") as file:
        for line in file:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue  # Skip malformed lines
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
    Read file and save each line as a list, stored in a larger list
    :param file_path: File path
    :return: List containing all lines, each line is also a list
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
    :param file_path: File path
    :return: List containing all results, each row is also a list
    """
    result = []
    with open(file_path, "r") as file:
        for line in file:
            parts = line.strip().split("\t")
            if len(parts) < 4:
                continue  # Skip malformed lines
            outcome = parts[0]
            steps = int(
                float(parts[1].strip("[]"))
            )  # Convert string '[180]' to list [180]
            score = int(parts[2])
            penalty = int(parts[3])
            result.append([outcome, steps, score, penalty])
    return result


def calculate_distance_matrix(reverse_dict, custom_distance, secondary_bk_trees):
    """
    Calculate distance matrix
    :param reverse_dict: Reverse dictionary
    :param custom_distance: Custom distance calculation function
    :param secondary_bk_trees: Dictionary of secondary BKTrees
    :return: Distance matrix
    """
    # Get the number of all clusters
    num_clusters = len(reverse_dict)

    # Initialize distance matrix
    distance_matrix = np.zeros((num_clusters, num_clusters))

    # Get states of all clusters
    clusters = list(reverse_dict.values())

    # Initialize last output time
    last_output_time = time.time()

    # Initialize progress threshold
    progress_threshold = 0.01  # 10%

    # Calculate distance between each pair of clusters
    for i in range(num_clusters):
        for j in range(
            i + 1, num_clusters
        ):  # Start from i+1 to avoid duplicate calculations
            state1 = clusters[i]["cluster"]
            state2 = clusters[j]["cluster"]

            # Get nodes for both states
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

    # Diagonal distances are 0
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

    # Initialize last output time
    last_output_time = time.time()

    # Initialize progress threshold
    progress_threshold = 0.01  # 1%

    # Calculate DTW distance between each pair of sequences
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
    :param file_path: File path
    """
    np.save(file_path, matrix)


def load_distance_matrix(file_path):
    """
    Load distance matrix from file
    :param file_path: File path
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
    :param secondary_bk_trees: Dictionary of secondary BKTrees
    :param distance_matrix_folder: Distance matrix save folder path
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
    :param dtw_distance_matrix_folder: DTW distance matrix save folder path
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
    Plot fitness landscape
    :param log_distance_matrix: Distance matrix between logs
    :param log_object: Fitness value of each log
    :param game_results: Game results list
    :param top_n: Select top_n logs for plotting, if None, plot all data
    """
    # If top_n is None, plot all data
    if top_n is None:
        top_n = len(log_object)

    # Take only the first top_n logs' distance matrix and fitness values
    log_distance_matrix = log_distance_matrix[:top_n, :top_n]
    log_object = log_object[:top_n]
    game_results = game_results[:top_n]

    # Use MDS to map logs to 2D plane
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
    log_positions = mds.fit_transform(log_distance_matrix)

    # Prepare grid for interpolation
    grid_x, grid_y = np.mgrid[
        log_positions[:, 0].min() : log_positions[:, 0].max() : 100j,
        log_positions[:, 1].min() : log_positions[:, 1].max() : 100j,
    ]

    # Use griddata for interpolation
    grid_z = griddata(
        points=log_positions,
        values=log_object,
        xi=(grid_x, grid_y),
        method="linear",  # Options: 'linear', 'nearest', 'cubic'
    )

    # Plot interpolated landscape
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
    Plot fitness landscape
    :param log_distance_matrix: Distance matrix between logs
    :param log_object: Fitness value of each log
    :param game_results: Game results list
    :param top_n: Select top_n logs for plotting, if None, plot all data
    """
    # If top_n is None, plot all data
    if top_n is None:
        top_n = len(log_object)

    # Take only the first top_n logs' distance matrix and fitness values
    log_distance_matrix = log_distance_matrix[:top_n, :top_n]
    log_object = log_object[:top_n]
    game_results = game_results[:top_n]

    # Use MDS to map logs to 2D plane
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
    log_positions = mds.fit_transform(log_distance_matrix)

    # Prepare grid for interpolation
    grid_x, grid_y = np.mgrid[
        log_positions[:, 0].min() : log_positions[:, 0].max() : 100j,
        log_positions[:, 1].min() : log_positions[:, 1].max() : 100j,
    ]

    # Use griddata for interpolation
    grid_z = griddata(
        points=log_positions,
        values=log_object,
        xi=(grid_x, grid_y),
        method="linear",  # Options: 'linear', 'nearest', 'cubic'
    )

    # Calculate position of 0 in color bar
    min_val = np.min(log_object)
    max_val = np.max(log_object)
    range_val = max_val - min_val

    if min_val >= 0 or max_val <= 0:
        # If all positive or all negative, set 0 at the middle of color bar
        zero_position = 0.5
    else:
        # Calculate the ratio of 0 between positive and negative values
        zero_position = (0 - min_val) / range_val

    # Custom color map, fix 0 value at specified position, set to white
    custom_colorscale = [
        [0, "#5c7ee6"],  # Minimum value corresponds to blue
        [zero_position, "#ebebeb"],  # 0 value corresponds to white
        [1, "#b62d0a"],  # Maximum value corresponds to red
    ]

    # Plot interpolated 3D landscape
    fig = go.Figure()

    # Add 3D landscape
    fig.add_trace(
        go.Surface(
            x=grid_x,
            y=grid_y,
            z=grid_z,
            colorscale=custom_colorscale,
            showscale=True,
            showlegend=False,  # Don't show legend
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
        visible=True,  # Show by default
    )
    fig.add_trace(sample_points)

    # Add global optimum red points
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
                visible=False,  # Don't show by default
                showlegend=False,  # Don't show legend
            )
        )
    fig.add_traces(global_optimum_points)

    # Add buttons to toggle display mode
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

    # Calculate max and min of log_positions
    x_min, x_max = log_positions[:, 0].min(), log_positions[:, 0].max()
    y_min, y_max = log_positions[:, 1].min(), log_positions[:, 1].max()
    # Calculate max and min of log_object
    z_min, z_max = min(log_object), max(log_object)

    # Update layout, dynamically set x and y ranges
    fig.update_layout(
        title="Fitness Landscape",
        font=dict(
            family="Times New Roman", size=28
        ),  # Set font to Times New Roman, size 16
        scene=dict(
            xaxis_title="",
            yaxis_title="",
            zaxis_title="",
            aspectmode="manual",  # Manually set ratio
            aspectratio=dict(x=1, y=1, z=0.75),  # Adjust Z axis ratio
            xaxis_range=[x_min - 5, x_max + 5],  # Dynamically set x axis range
            yaxis_range=[y_min - 5, y_max + 5],  # Dynamically set y axis range
            zaxis_range=[z_min - 3, z_max + 3],  # Dynamically set z axis range
            xaxis=dict(
                tickvals=[],  # Hide X axis tick values
                backgroundcolor="white",  # Set X axis background color to white
                gridcolor="white",  # Set X axis grid color to white
                linecolor="white",  # Set X axis line color to white
            ),
            yaxis=dict(
                tickvals=[],  # Hide Y axis tick values
                backgroundcolor="white",  # Set Y axis background color to white
                gridcolor="white",  # Set Y axis grid color to white
                linecolor="white",  # Set Y axis line color to white
            ),
            zaxis=dict(
                tickvals=[],  # Hide Z axis tick values
                backgroundcolor="white",  # Set Z axis background color to white
                gridcolor="white",  # Set Z axis grid color to white
                linecolor="white",  # Set Z axis line color to white
            ),
            bgcolor="white",  # Set scene background color to white
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
    Get all distances between states in optimal solutions, and distances between optimal solution sequences
    :param state_distance_matrix: Distance matrix between states
    :param log_distance_matrix: Distance matrix between log sequences
    :param log_object: Fitness value of each log
    :param game_results: Game results list
    :param top_n: Select top_n logs for calculation, if None, use all data
    :return: Distance matrix between optimal solution states, distance matrix between optimal solution sequences
    """
    # If top_n is None, use all data
    if top_n is None:
        top_n = len(log_object)

    # Take only the first top_n logs' fitness values and game results
    log_object = log_object[:top_n]
    game_results = game_results[:top_n]

    # Find optimal solution indices
    max_fitness = np.max(log_object)
    optimal_indices = np.where(log_object == max_fitness)[0]

    # Extract log sequences and results corresponding to optimal solutions
    optimal_logs = [state_log[i] for i in optimal_indices]
    optimal_logs_result = [game_results[i] for i in optimal_indices]

    # Extract all states in optimal solution sequences
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


def plot_state_transition_graph(
    optimal_state_distance_matrix,
    optimal_log_distance_matrix,
    optimal_logs,
    optimal_states,
    k=3,
    width=350,
    height=600,
):
    """
    Use Plotly to plot state transition graph between states, and save as image
    :param optimal_state_distance_matrix: Distance matrix between optimal solution states
    :param optimal_log_distance_matrix: Distance matrix between log sequences
    :param optimal_logs: Log sequences corresponding to optimal solutions
    :param optimal_states: State list in optimal solutions
    :param k: Number of clusters
    """
    # Get number of states
    num_states = len(optimal_states)

    # Use MDS (Multi-Dimensional Scaling) to distribute states on 2D plane according to distance relationships
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

    # Create directed edges for state transitions (main graph)
    edge_x = []
    edge_y = []
    drawn_edges = set()  # Used to record already drawn edges

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
            showticklabels=False,  # Don't show x axis ticks
            showgrid=False,  # Don't show x axis grid
            zeroline=False,  # Don't show x axis zero line
            visible=False,  # Don't show x axis
        ),
        yaxis=dict(
            showticklabels=False,  # Don't show y axis ticks
            showgrid=False,  # Don't show y axis grid
            zeroline=False,  # Don't show y axis zero line
            visible=False,  # Don't show y axis
        ),
        plot_bgcolor="white",  # Set plot area background color to white
        paper_bgcolor="white",  # Set entire chart background color to white
    )

    # Show figure
    fig_main.show()

    # Save main graph as image, specify width and height
    pio.write_image(
        fig_main,
        f"{distance_matrix_folder}all_state_transition_graph.pdf",
        width=width,
        height=height,
        scale=4,
    )

    # Cluster logs
    kmeans = KMeans(n_clusters=k, random_state=42)
    log_clusters = kmeans.fit_predict(optimal_log_distance_matrix)

    # Draw a graph for each cluster
    for cluster_id in range(k):
        cluster_logs = [
            optimal_logs[i]
            for i in range(len(optimal_logs))
            if log_clusters[i] == cluster_id
        ]

        # Create directed edges for state transitions (cluster graph)
        edge_x_cluster = []
        edge_y_cluster = []
        edge_x_other = []
        edge_y_other = []

        # Record already drawn edges
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

        # Draw other undrawn edges
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
            line=dict(width=2, color=f"{COLOR_LIST[cluster_id]}"),
            hoverinfo="none",
        )

        state_edges_other = go.Scatter(
            x=edge_x_other,
            y=edge_y_other,
            mode="lines",
            line=dict(width=2, color="black"),
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
                showticklabels=False,  # Don't show x axis ticks
                showgrid=False,  # Don't show x axis grid
                zeroline=False,  # Don't show x axis zero line
                visible=False,  # Don't show x axis
            ),
            yaxis=dict(
                showticklabels=False,  # Don't show y axis ticks
                showgrid=False,  # Don't show y axis grid
                zeroline=False,  # Don't show y axis zero line
                visible=False,  # Don't show y axis
            ),
            plot_bgcolor="white",  # Set plot area background color to white
            paper_bgcolor="white",  # Set entire chart background color to white
        )

        # Show cluster graph
        fig_cluster.show()

        # Save cluster graph as image, specify width and height
        pio.write_image(
            fig_cluster,
            f"{distance_matrix_folder}cluster_{cluster_id + 1}_state_transition_graph.pdf",
            width=width,
            height=height,
            scale=4,
        )


# Example usage
if __name__ == "__main__":
    # Load BKTree
    primary_bk_tree = load_bk_tree_from_file(primary_bktree_path)
    secondary_bk_trees = {}

    print("Root cluster ID:", primary_bk_tree.root.cluster_id)
    print("Root state:", primary_bk_tree.root.state)
    print("Root children:", primary_bk_tree.root.children.keys())
    cluster_count = get_max_cluster_id(primary_bk_tree)
    print("Cluster count:", cluster_count)

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
    # Calculate and save distance matrix
    state_distance_matrix = calculate_and_save_distance_matrix(
        reverse_dict, custom_distance, secondary_bk_trees, distance_matrix_folder
    )

    # Calculate and save DTW distance matrix
    log_distance_matrix = calculate_and_save_dtw_distance_matrix(
        state_log, state_distance_matrix, distance_matrix_folder
    )

    print("########################################################################")
    game_results = read_game_result_file(game_result_path)
    log_object = [log[2] + log[3] for log in game_results]

    # Plot fitness landscape
    plot_fitness_landscape_3d(log_distance_matrix, log_object, game_results, top_n=None)

    # # Get optimal solution related distance matrices
    # optimal_state_distance_matrix, optimal_log_distance_matrix, optimal_indices, optimal_logs, optimal_states = get_optimal_solution_distances(
    #     state_distance_matrix, log_distance_matrix, log_object, game_results, state_log
    # )
    #
    # plot_state_transition_graph(optimal_state_distance_matrix, optimal_log_distance_matrix, optimal_logs, optimal_states, k=3, width=300, height=600)

    # # Print or save results
    # print("Optimal State Distance Matrix:")
    # print(optimal_state_distance_matrix)
    # print("Optimal Log Distance Matrix:")
    # print(optimal_log_distance_matrix)
    # print("Optimal Logs:")
    # for log in optimal_logs:
    #     print(log)

    print("########################################################################")

    # plot_fitness_landscape_3d_matplot(log_distance_matrix, log_object, game_results, top_n=50)
