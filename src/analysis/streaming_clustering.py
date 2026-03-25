"""
Streaming Clustering Algorithm Implementation

Contains implementation and comparative analysis of streaming clustering algorithms including BKTree, DenStream, etc.
"""

import json
import random

import numpy as np
from collections import defaultdict
import time
from matplotlib import pyplot as plt
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn.manifold import MDS, TSNE
from scipy.spatial import ConvexHull
from matplotlib.patches import Circle

# Use project's internal distance calculation module
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.distance.base import CustomDistance
from config import get_multi_alg_path, get_multi_alg_output_path, DATA_DIR

# Dataset configuration
sce_id = "sce-3m"
path_data = get_multi_alg_path(sce_id, "data.json")

# Streaming clustering algorithm configuration
# multi_alg = ['BKTree']
# multi_alg = ['BKTreeInt']
# multi_alg = ['BKTreeFine']
multi_alg = ["DenStream"]
custom_distance = CustomDistance(threshold=0.5)

# Global variable, controls whether to calculate clustering evaluation metrics
CALCULATE_METRICS = False


def load_dataset(path):
    with open(path, "r") as file:
        state_result = json.load(file)

    new_state_result = {}
    for key, value in state_result.items():
        alias = value.get("alias")
        if alias is not None:
            new_key = alias
            new_state_result[new_key] = value
        else:
            print(f"Warning: Key '{key}' does not have an alias. Skipping...")
    return new_state_result


# ===============================================================================
# BKTree
class ClusterNode:
    def __init__(self, state, cluster_id):
        self.state = state
        self.cluster_id = cluster_id
        self.children = {}

    def add_child(self, distance, child):
        self.children[distance] = child


class BKTree:
    def __init__(self, distance_func):
        self.root = None
        self.distance_func = distance_func
        self.next_cluster_id = 2  # Initialize cluster ID counter

    def insert(self, node, parent=None):
        if parent is None:
            self.root = node
            node.cluster_id = self.next_cluster_id  # Assign cluster ID
            self.next_cluster_id += 1  # Update cluster ID counter
            return
        dist = self.distance_func(node.state, parent.state)[0]  # Get distance value
        if dist in parent.children:
            self.insert(node, parent.children[dist])
        else:
            parent.add_child(dist, node)
            node.cluster_id = self.next_cluster_id  # Assign cluster ID
            self.next_cluster_id += 1  # Update cluster ID counter

    def query(self, state, threshold):
        def search(node, dist):
            if dist < threshold:
                return node.cluster_id
            for d, child in node.children.items():
                if abs(d - dist) < threshold:
                    result = search(child, self.distance_func(state, child.state)[0])
                    if result is not None:
                        return result
            return None

        if self.root is None:
            return None
        return search(self.root, self.distance_func(state, self.root.state)[0])

    def get_next_cluster_id(self):
        """Get next cluster ID"""
        return self.next_cluster_id


def classify_new_state(new_state, bktree, threshold=1.0):
    cluster_id = bktree.query(new_state, threshold)
    if cluster_id is not None:
        return cluster_id
    else:
        new_cluster_id = bktree.get_next_cluster_id()
        new_node = ClusterNode(new_state, new_cluster_id)
        bktree.insert(new_node, bktree.root)
        return new_cluster_id


def traverse_tree(node):
    nodes = [node]
    for child in node.children.values():
        nodes.extend(traverse_tree(child))
    return nodes


# ===============================================================================
# BKTreeInt
class BKTreeInt:
    def __init__(self, distance_func):
        self.root = None
        self.distance_func = distance_func
        self.next_cluster_id = 2  # Initialize cluster ID counter

    def insert(self, node, parent=None):
        if parent is None:
            self.root = node
            node.cluster_id = self.next_cluster_id  # Assign cluster ID
            self.next_cluster_id += 1  # Update cluster ID counter
            return
        dist = self.round_distance(
            self.distance_func(node.state, parent.state)[0]
        )  # Round and ensure minimum is 1
        if dist in parent.children:
            self.insert(node, parent.children[dist])
        else:
            parent.add_child(dist, node)
            node.cluster_id = self.next_cluster_id  # Assign cluster ID
            self.next_cluster_id += 1  # Update cluster ID counter

    def query(self, state, threshold):
        def search(node, dist):
            if dist < threshold:
                return node.cluster_id
            for d, child in node.children.items():
                if abs(d - dist) < threshold:
                    result = search(
                        child,
                        self.round_distance(self.distance_func(state, child.state)[0]),
                    )  # Round and ensure minimum is 1
                    if result is not None:
                        return result
            return None

        if self.root is None:
            return None
        return search(
            self.root,
            self.round_distance(self.distance_func(state, self.root.state)[0]),
        )  # Round and ensure minimum is 1

    def round_distance(self, distance):
        """Round and ensure distance is at least 1"""
        return round(distance)

    def get_next_cluster_id(self):
        """Get next cluster ID"""
        return self.next_cluster_id


# ===============================================================================
# ===============================================================================
# BKTreeFine
class BKTreeFine:
    def __init__(self, distance_func, distance_index=0):
        self.root = None
        self.distance_func = distance_func
        self.distance_index = distance_index
        self.next_cluster_id = 2

    def insert(self, node, parent=None):
        if parent is None:
            self.root = node
            node.cluster_id = self.next_cluster_id
            self.next_cluster_id += 1
            return
        dist = self.round_distance(
            self.distance_func(node.state, parent.state)[self.distance_index]
        )
        if dist in parent.children:
            self.insert(node, parent.children[dist])
        else:
            parent.add_child(dist, node)
            node.cluster_id = self.next_cluster_id
            self.next_cluster_id += 1

    def query(self, state, threshold):
        def search(node, dist):
            if dist < threshold:
                return node.cluster_id
            for d, child in node.children.items():
                if abs(d - dist) < threshold:
                    result = search(
                        child,
                        self.round_distance(
                            self.distance_func(state, child.state)[self.distance_index]
                        ),
                    )
                    if result is not None:
                        return result
            return None

        if self.root is None:
            return None
        return search(
            self.root,
            self.round_distance(
                self.distance_func(state, self.root.state)[self.distance_index]
            ),
        )

    def round_distance(self, distance):
        """Round and ensure distance is at least 1"""
        return round(distance)

    def get_next_cluster_id(self):
        return self.next_cluster_id


# ===============================================================================
# ===============================================================================
# DenStream
class MicroCluster:
    def __init__(self, state, timestamp, decay_factor):
        self.state = state
        self.timestamp = timestamp
        self.decay_factor = decay_factor
        self.weight = 1.0  # Initial weight is 1

    def update(self, new_state, timestamp):
        self.weight *= self.decay_factor  # Weight decay
        self.weight += 1.0  # New data point weight
        self.timestamp = timestamp

    def radius(self):
        # Calculate micro-cluster radius (simplified as function of weight)
        return self.weight**0.5


class DenStream:
    def __init__(self, distance_func, decay_factor, beta, mu, epsilon):
        self.distance_func = distance_func
        self.decay_factor = decay_factor
        self.beta = beta
        self.mu = mu
        self.epsilon = epsilon
        self.core_micro_clusters = []
        self.outlier_micro_clusters = []
        self.timestamp = 0

    def insert(self, state):
        self.timestamp += 1
        # Check if new state can be assigned to existing core micro-cluster
        assigned = False
        for cluster in self.core_micro_clusters:
            if self.distance_func(state, cluster.state)[0] <= self.epsilon:
                cluster.update(state, self.timestamp)
                assigned = True
                break
        if not assigned:
            # If cannot assign to core micro-cluster, try assigning to outlier micro-cluster
            assigned = False
            for cluster in self.outlier_micro_clusters:
                if self.distance_func(state, cluster.state)[0] <= self.epsilon:
                    cluster.update(state, self.timestamp)
                    assigned = True
                    # Check if should upgrade to core micro-cluster
                    if cluster.weight >= self.mu:
                        self.core_micro_clusters.append(cluster)
                        self.outlier_micro_clusters.remove(cluster)
                    break
            if not assigned:
                # If cannot assign to any micro-cluster, create new outlier micro-cluster
                new_cluster = MicroCluster(state, self.timestamp, self.decay_factor)
                self.outlier_micro_clusters.append(new_cluster)

        # Clean up expired micro-clusters
        self._prune_micro_clusters()

    def _prune_micro_clusters(self):
        # Clean up expired micro-clusters
        current_time = self.timestamp
        self.core_micro_clusters = [
            cluster for cluster in self.core_micro_clusters if cluster.weight >= self.mu
        ]
        self.outlier_micro_clusters = [
            cluster
            for cluster in self.outlier_micro_clusters
            if cluster.weight >= self.beta
        ]

    def get_clusters(self):
        # Return current core micro-clusters
        return self.core_micro_clusters


# ===============================================================================
# ===============================================================================
# XXXXXX

# ===============================================================================
# ===============================================================================
# XXXXXX

# ===============================================================================


def calculate_cluster_centroid(states):
    """
    Calculate cluster centroid for given state list.
    Assumes states is a 2D array, each row represents a state's feature vector.
    """
    if not states:
        return None
    state_list = [state["state"][0] for state in states]
    # Initialize a dictionary to store averages
    average_dict = {"blue_army": [], "red_army": []}
    # Get length of 'blue_army' and 'red_army' 2D lists in first dictionary
    num_blue = len(state_list[0]["blue_army"])
    num_red = len(state_list[0]["red_army"])

    # Filter out dictionaries with incorrect lengths
    filtered_state_list = [
        d
        for d in state_list
        if len(d["blue_army"]) == num_blue and len(d["red_army"]) == num_red
    ]

    # If filtered list is empty, return empty average_dict directly
    if not filtered_state_list:
        print("Filtered list is empty, cannot calculate average")
    else:
        # Iterate through each position in 'blue_army'
        for i in range(num_blue):
            # Initialize sum list for each position
            blue_sum = [0, 0, 0]
            for d in filtered_state_list:
                # Iterate through each dictionary in list, add corresponding position values to sum list
                blue_sum = [x + y for x, y in zip(blue_sum, d["blue_army"][i])]
            # Calculate average and add to average_dict
            average_dict["blue_army"].append(
                [x / len(filtered_state_list) for x in blue_sum]
            )

        # Iterate through each position in 'red_army'
        for i in range(num_red):
            red_sum = [0, 0, 0]
            for d in filtered_state_list:
                red_sum = [x + y for x, y in zip(red_sum, d["red_army"][i])]
            average_dict["red_army"].append(
                [x / len(filtered_state_list) for x in red_sum]
            )

    centroid = {
        "state": [
            {
                "blue_army": average_dict["blue_army"],
                "red_army": average_dict["red_army"],
            }
        ]
    }
    return centroid


def calculate_clustering_metrics(states, clusters, distance_func, sample_size=100):
    cluster_representatives = defaultdict(list)
    for state, cluster in zip(states, clusters):
        cluster_representatives[cluster].append(state)

    # Calculate centroid for each cluster
    cluster_centroids = {}
    total_clusters = len(cluster_representatives)
    progress_threshold = max(
        1, total_clusters // 10
    )  # Output progress every 10% of clusters
    # Initialize total start time
    total_start_time = time.time()
    last_output_time = total_start_time

    def output_progress(index, total, last_output_time, total_start_time, message=""):
        current_time = time.time()
        time_elapsed = current_time - last_output_time
        total_elapsed_time = (
            current_time - total_start_time
        )  # Calculate total elapsed time
        print(
            f"{message} {index + 1} out of {total} clusters ({(index + 1) / total * 100:.1f}%) "
            f"(Time elapsed: {time_elapsed:.2f} seconds, Total time: {total_elapsed_time:.2f} seconds)"
        )
        return current_time

    for index, (cluster_id, states) in enumerate(cluster_representatives.items()):
        # Calculate centroid
        cluster_centroids[cluster_id] = calculate_cluster_centroid(states)

        # # Output progress
        if (index + 1) % progress_threshold == 0 or index == total_clusters - 1:
            last_output_time = output_progress(
                index,
                total_clusters,
                last_output_time,
                total_start_time,
                message="Calculated centroid for",
            )

    # Initialize silhouette coefficient list
    silhouette_scores = []

    # Iterate through each cluster
    # print("Calculating silhouette scores for clusters...")
    for index, (cluster_id, states) in enumerate(cluster_representatives.items()):
        # If log count is less than sample size, use all logs directly
        sample_states = random.sample(states, min(sample_size, len(states)))

        # Calculate intra-cluster distance
        centroid = cluster_centroids[cluster_id]
        a = np.mean([distance_func(state, centroid)[0] for state in sample_states])

        # Calculate inter-cluster distance
        other_clusters = [
            c_id for c_id in cluster_representatives.keys() if c_id != cluster_id
        ]
        b = np.min(
            [
                np.mean(
                    [
                        distance_func(state, cluster_centroids[other_cluster])[0]
                        for state in sample_states
                    ]
                )
                for other_cluster in other_clusters
            ]
        )

        # Calculate silhouette coefficient
        silhouette = (b - a) / max(a, b) if max(a, b) > 0 else 0
        silhouette_scores.append(silhouette)

        progress_threshold = max(
            1, total_clusters // 100
        )  # Output progress every 10% of clusters
        # Output progress
        if (index + 1) % progress_threshold == 0 or index == total_clusters - 1:
            last_output_time = output_progress(
                index,
                total_clusters,
                last_output_time,
                total_start_time,
                message="Processed",
            )

    # Return average silhouette coefficient
    return round(np.mean(silhouette_scores), 3)


def calculate_clustering_metrics_fine(states, clusters, distance_func, sample_size=100):
    cluster_representatives = defaultdict(list)
    for state, cluster in zip(states, clusters):
        cluster_representatives[cluster].append(state)

    # Calculate centroid for each cluster
    cluster_centroids = {}
    total_clusters = len(cluster_representatives)
    progress_threshold = max(
        1, total_clusters // 10
    )  # Output progress every 10% of clusters
    # Initialize total start time
    total_start_time = time.time()
    last_output_time = total_start_time

    def output_progress(index, total, last_output_time, total_start_time, message=""):
        current_time = time.time()
        time_elapsed = current_time - last_output_time
        total_elapsed_time = (
            current_time - total_start_time
        )  # Calculate total elapsed time
        print(
            f"{message} {index + 1} out of {total} clusters ({(index + 1) / total * 100:.1f}%) "
            f"(Time elapsed: {time_elapsed:.2f} seconds, Total time: {total_elapsed_time:.2f} seconds)"
        )
        return current_time

    for index, (cluster_id, states) in enumerate(cluster_representatives.items()):
        # Calculate centroid
        cluster_centroids[cluster_id] = calculate_cluster_centroid(states)

        # # Output progress
        if (index + 1) % progress_threshold == 0 or index == total_clusters - 1:
            last_output_time = output_progress(
                index,
                total_clusters,
                last_output_time,
                total_start_time,
                message="Calculated centroid for",
            )

    # Initialize silhouette coefficient list
    silhouette_scores = []

    # Iterate through each cluster
    # print("Calculating silhouette scores for clusters...")
    for index, (cluster_id, states) in enumerate(cluster_representatives.items()):
        # If log count is less than sample size, use all logs directly
        sample_states = random.sample(states, min(sample_size, len(states)))

        # Calculate intra-cluster distance
        centroid = cluster_centroids[cluster_id]
        a = np.mean(
            [
                np.sqrt(
                    np.sum(
                        distance_func(state, centroid)[0] ** 2
                        + distance_func(state, centroid)[1] ** 2
                    )
                )
                for state in sample_states
            ]
        )

        # Calculate inter-cluster distance
        other_clusters = [
            c_id for c_id in cluster_representatives.keys() if c_id != cluster_id
        ]
        b = np.min(
            [
                np.mean(
                    [
                        np.sqrt(
                            np.sum(
                                distance_func(state, cluster_centroids[other_cluster])[
                                    0
                                ]
                                ** 2
                                + distance_func(
                                    state, cluster_centroids[other_cluster]
                                )[1]
                                ** 2
                            )
                        )
                        for state in sample_states
                    ]
                )
                for other_cluster in other_clusters
            ]
        )

        # Calculate silhouette coefficient
        silhouette = (b - a) / max(a, b) if max(a, b) > 0 else 0
        silhouette_scores.append(silhouette)

        progress_threshold = max(
            1, total_clusters // 100
        )  # Output progress every 10% of clusters
        # Output progress
        if (index + 1) % progress_threshold == 0 or index == total_clusters - 1:
            last_output_time = output_progress(
                index,
                total_clusters,
                last_output_time,
                total_start_time,
                message="Processed",
            )

    # Return average silhouette coefficient
    return round(np.mean(silhouette_scores), 3)


def calculate_clustering_metrics_clear(
    states, clusters, distance_func, sample_size=100
):
    cluster_representatives = defaultdict(list)
    for state, cluster in zip(states, clusters):
        cluster_representatives[cluster].append(state)

    # Calculate centroid for each cluster
    cluster_centroids = {}
    total_clusters = len(cluster_representatives)
    progress_threshold = max(
        1, total_clusters // 10
    )  # Output progress every 10% of clusters
    # Initialize total start time
    total_start_time = time.time()
    last_output_time = total_start_time

    for index, (cluster_id, states) in enumerate(cluster_representatives.items()):
        # Calculate centroid
        cluster_centroids[cluster_id] = calculate_cluster_centroid(states)

    # Initialize silhouette coefficient list
    silhouette_scores = []

    # Iterate through each cluster
    # print("Calculating silhouette scores for clusters...")
    for index, (cluster_id, states) in enumerate(cluster_representatives.items()):
        # If log count is less than sample size, use all logs directly
        sample_states = random.sample(states, min(sample_size, len(states)))

        # Calculate intra-cluster distance
        centroid = cluster_centroids[cluster_id]
        a = np.mean([distance_func(state, centroid)[0] for state in sample_states])

        # Calculate inter-cluster distance
        other_clusters = [
            c_id for c_id in cluster_representatives.keys() if c_id != cluster_id
        ]
        b = np.min(
            [
                np.mean(
                    [
                        distance_func(state, cluster_centroids[other_cluster])[0]
                        for state in sample_states
                    ]
                )
                for other_cluster in other_clusters
            ]
        )

        # Calculate silhouette coefficient
        silhouette = (b - a) / max(a, b) if max(a, b) > 0 else 0
        silhouette_scores.append(silhouette)

    # Return average silhouette coefficient
    return round(np.mean(silhouette_scores), 3)


def save_results_to_file(file_path, results):
    with open(file_path, "w") as file:
        for result in results:
            file.write(result + "\n")


def main():
    print(f"Loading dataset from {path_data}...")
    data = load_dataset(path_data)
    all_state_ids = list(data.keys())
    states = list(data.values())
    total_states = len(states)
    print(f"Total states loaded: {total_states}")

    for alg in multi_alg:
        if alg == "BKTree":
            print(f"Starting clustering with {alg}...")

            # thresholds = [0.1, 0.15, 0.2, 0.25, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.1, 1.2, 1.25, 1.3, 1.4, 1.5, 2.5, 5.0]
            thresholds = [0.55, 0.5, 0.25, 0.2, 0.15, 0.1]
            # thresholds = [1.1,1.2,1.25,1.3,1.4]
            for threshold in thresholds:
                print(f"Processing with threshold {threshold}...")

                # Initialize list to store 5 run results
                elapsed_times = []
                cluster_counts = []
                silhouette_avgs = []

                for run in range(1):
                    # print(f"Run {run + 1} with threshold {threshold}...")

                    clusters = np.zeros(total_states, dtype=int)
                    cluster_dict = defaultdict(list)

                    # Initialize BKTree
                    bktree = BKTree(custom_distance.multi_distance)

                    # Insert first state as root node
                    root = ClusterNode(states[0], 1)
                    bktree.root = root
                    clusters[0] = 1
                    cluster_dict[1].append(all_state_ids[0])

                    start_time = time.time()

                    for i in range(1, total_states):
                        new_cluster_id = classify_new_state(
                            states[i], bktree, threshold=threshold
                        )
                        clusters[i] = new_cluster_id
                        cluster_dict[new_cluster_id].append(all_state_ids[i])

                    elapsed_time = (
                        time.time() - start_time
                    )  # Total elapsed time for current run
                    elapsed_times.append(elapsed_time)
                    cluster_counts.append(len(cluster_dict))

                    print(
                        f"threshold {threshold} completed in {elapsed_time:.3f} seconds."
                    )
                    print(f"Number of clusters in {threshold}: {len(cluster_dict)}")

                    # Calculate silhouette coefficient after processing all data
                    silhouette_avg = calculate_clustering_metrics(
                        states,
                        clusters,
                        custom_distance.multi_distance,
                        sample_size=100,
                    )
                    silhouette_avgs.append(silhouette_avg)

                # Calculate averages
                avg_elapsed_time = sum(elapsed_times) / len(elapsed_times)
                avg_cluster_count = sum(cluster_counts) / len(cluster_counts)
                avg_silhouette_avg = sum(silhouette_avgs) / len(silhouette_avgs)

                # Output results
                print(f"Threshold: {threshold}")
                print(f"Average Elapsed Time: {avg_elapsed_time:.3f} seconds")
                print(f"Average Number of Clusters: {avg_cluster_count:.2f}")
                print(f"Average Silhouette Coefficient: {avg_silhouette_avg:.3f}")
                print("-" * 50)
        elif alg == "BKTreeInt":
            print(f"Starting clustering with {alg}...")

            # thresholds = [0.1, 0.15, 0.2, 0.25, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.1, 1.2, 1.25, 1.3, 1.4, 1.5, 2.5, 5.0]
            thresholds = [0.55, 0.5, 0.25, 0.2, 0.15, 0.1]
            # thresholds = [1.1,1.2,1.25,1.3,1.4]
            for threshold in thresholds:
                print(f"Processing with threshold {threshold}...")

                # Initialize list to store 5 run results
                elapsed_times = []
                cluster_counts = []
                silhouette_avgs = []

                for run in range(1):
                    # print(f"Run {run + 1} with threshold {threshold}...")

                    clusters = np.zeros(total_states, dtype=int)
                    cluster_dict = defaultdict(list)

                    # Initialize BKTreeInt
                    bktree = BKTreeInt(custom_distance.multi_distance)

                    # Insert first state as root node
                    root = ClusterNode(states[0], 1)
                    bktree.root = root
                    clusters[0] = 1
                    cluster_dict[1].append(all_state_ids[0])

                    start_time = time.time()

                    for i in range(1, total_states):
                        new_cluster_id = classify_new_state(
                            states[i], bktree, threshold=threshold
                        )
                        clusters[i] = new_cluster_id
                        cluster_dict[new_cluster_id].append(all_state_ids[i])

                    elapsed_time = (
                        time.time() - start_time
                    )  # Total elapsed time for current run
                    elapsed_times.append(elapsed_time)
                    cluster_counts.append(len(cluster_dict))

                    print(
                        f"threshold {threshold} completed in {elapsed_time:.3f} seconds."
                    )
                    print(f"Number of clusters in {threshold}: {len(cluster_dict)}")

                    # Calculate silhouette coefficient after processing all data
                    silhouette_avg = calculate_clustering_metrics(
                        states,
                        clusters,
                        custom_distance.multi_distance,
                        sample_size=100,
                    )
                    silhouette_avgs.append(silhouette_avg)

                # Calculate averages
                avg_elapsed_time = sum(elapsed_times) / len(elapsed_times)
                avg_cluster_count = sum(cluster_counts) / len(cluster_counts)
                avg_silhouette_avg = sum(silhouette_avgs) / len(silhouette_avgs)

                # Output results
                print(f"Threshold: {threshold}")
                print(f"Average Elapsed Time: {avg_elapsed_time:.3f} seconds")
                print(f"Average Number of Clusters: {avg_cluster_count:.2f}")
                print(f"Average Silhouette Coefficient: {avg_silhouette_avg:.3f}")
                print("-" * 50)
        elif alg == "BKTreeFine":
            print(f"Starting clustering with {alg}...")

            # thresholds = [0.1, 0.15, 0.2, 0.25, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.1, 1.2, 1.25, 1.3, 1.4, 1.5, 2.5, 5.0]
            thresholds = [1.0]
            # thresholds = [1.1,1.2,1.25,1.3,1.4]
            for threshold in thresholds:
                print(f"Processing with threshold {threshold}...")

                # Initialize list to store 5 run results
                elapsed_times = []
                cluster_counts = []
                silhouette_avgs = []

                for run in range(1):
                    # print(f"Run {run + 1} with threshold {threshold}...")

                    clusters = np.zeros(total_states, dtype=tuple)
                    cluster_dict = defaultdict(list)

                    start_time = time.time()

                    custom_distance_manager = CustomDistance(threshold=0.5)
                    primary_bktree = BKTreeFine(
                        custom_distance_manager.multi_distance, distance_index=0
                    )
                    secondary_bktree = defaultdict(
                        lambda: BKTreeFine(
                            custom_distance_manager.multi_distance, distance_index=1
                        )
                    )

                    def get_state_cluster(norm_state):
                        if primary_bktree.root is None:
                            primary_bktree.root = ClusterNode(norm_state, 1)
                            secondary_bktree[1].root = ClusterNode(norm_state, 1)
                            # self.primary_bktree.root.state_list = [norm_state]
                            return (1, 1)
                        else:
                            new_cluster_id = classify_new_state(
                                norm_state, primary_bktree, threshold=5.0
                            )
                            if secondary_bktree[new_cluster_id].root is None:
                                secondary_bktree[new_cluster_id].root = ClusterNode(
                                    norm_state, 1
                                )
                                return (new_cluster_id, 1)
                            else:
                                new_sub_cluster_id = classify_new_state(
                                    norm_state,
                                    secondary_bktree[new_cluster_id],
                                    threshold=0.5,
                                )
                                return (new_cluster_id, new_sub_cluster_id)

                    # Calculate steps needed to process 1% of data
                    steps_per_percent = total_states / 10
                    last_reported_progress = -1

                    for i in range(0, total_states):
                        state_cluster_id = get_state_cluster(states[i])
                        clusters[i] = state_cluster_id
                        cluster_dict[state_cluster_id].append(all_state_ids[i])

                        # Calculate current progress percentage
                        current_progress = int((i / total_states) * 100)

                        # Output current total elapsed time every 10% of data processed
                        if (
                            current_progress % 10 == 0
                            and current_progress > last_reported_progress
                        ):
                            elapsed_time = (
                                time.time() - start_time
                            )  # Total elapsed time for current run
                            print(
                                f"Processed {current_progress}% of data, current total elapsed time: {elapsed_time:.2f} seconds"
                            )
                            elapsed_times.append(elapsed_time)
                            cluster_counts.append(len(cluster_dict))
                            last_reported_progress = current_progress

                    # Record final total elapsed time
                    elapsed_time = time.time() - start_time
                    elapsed_times.append(elapsed_time)
                    cluster_counts.append(len(cluster_dict))
                    print(
                        f"threshold {threshold} processing completed, total elapsed time: {elapsed_time:.3f} seconds"
                    )
                    print(f"Number of clusters in {threshold}: {len(cluster_dict)}")

                    # Calculate silhouette coefficient after processing all data
                    silhouette_avg = calculate_clustering_metrics_fine(
                        states,
                        clusters,
                        custom_distance_manager.multi_distance,
                        sample_size=100,
                    )
                    silhouette_avgs.append(silhouette_avg)

                # Calculate averages
                avg_elapsed_time = sum(elapsed_times) / len(elapsed_times)
                avg_cluster_count = sum(cluster_counts) / len(cluster_counts)
                avg_silhouette_avg = sum(silhouette_avgs) / len(silhouette_avgs)

                # Output results
                print(f"Threshold: {threshold}")
                print(f"Average Elapsed Time: {avg_elapsed_time:.3f} seconds")
                print(f"Average Number of Clusters: {avg_cluster_count:.2f}")
                print(f"Average Silhouette Coefficient: {avg_silhouette_avg:.3f}")
                print("-" * 50)
        elif alg == "DenStream":
            print(f"Starting clustering with {alg}...")

            # Define parameter ranges to test
            decay_factors = [0.1, 0.5, 1]
            betas = [0.1, 0.5, 1]
            mus = [0.1, 0.5, 1]
            epsilons = [0.1, 0.5, 1.0, 1.1, 1.2, 1.25, 1.3, 1.4, 1.5, 2.0, 2.5, 5.0]
            epsilons = [0.5, 1.0, 1.1, 1.2, 1.25, 1.3, 1.4, 1.5, 2.0, 2.5, 5.0]

            for decay_factor in decay_factors:
                for beta in betas:
                    for mu in mus:
                        for epsilon in epsilons:
                            print(
                                f"Processing with decay_factor={decay_factor}, beta={beta}, mu={mu}, epsilon={epsilon}..."
                            )

                            elapsed_times = []
                            cluster_counts = []
                            silhouette_avgs = []

                            for run in range(1):
                                # Initialize DenStream
                                denstream = DenStream(
                                    custom_distance.multi_distance,
                                    decay_factor,
                                    beta,
                                    mu,
                                    epsilon,
                                )

                                denstream_start_time = time.time()

                                # # Calculate steps needed to process 1% of data
                                # steps_per_percent = total_states / 10
                                # last_reported_progress = -1

                                for i in range(total_states):
                                    denstream.insert(states[i])

                                    # # Calculate current progress percentage
                                    # current_progress = int((i + 1) / steps_per_percent)
                                    #
                                    # # If current progress percentage is greater than last reported, output info
                                    # if current_progress > last_reported_progress:
                                    #     elapsed_time = time.time() - start_time  # Total elapsed time for current run
                                    #     print(
                                    #         f"Processed {current_progress}% of the data. Elapsed time: {elapsed_time:.2f} seconds.")
                                    #     last_reported_progress = current_progress

                                denstream_end_time = time.time()
                                denstream_elapsed_time = (
                                    denstream_end_time - denstream_start_time
                                )
                                # elapsed_times.append(elapsed_time)
                                core_clusters = denstream.get_clusters()
                                cluster_counts.append(len(core_clusters))

                                print(
                                    f"decay_factor: {decay_factor}, beta: {beta}, mu: {mu}, epsilon: {epsilon}"
                                )
                                # print(f"Elapsed Time: {elapsed_time:.3f} seconds")
                                print(f"Number of core clusters: {len(core_clusters)}")

                                # Calculate silhouette coefficient after processing all data
                                # Note: DenStream silhouette coefficient calculation needs to use micro-cluster centroid as cluster center
                                cluster_ids = []
                                cluster_assignment_start_time = (
                                    time.time()
                                )  # Cluster assignment start time
                                for i, state in enumerate(states):
                                    min_distance = float("inf")
                                    assigned_cluster_id = None
                                    for j, cluster in enumerate(core_clusters):
                                        distance = custom_distance.multi_distance(
                                            state, cluster.state
                                        )[0]
                                        if distance < 1:  # If distance is less than 1
                                            assigned_cluster_id = (
                                                j + 1
                                            )  # Assign cluster ID directly
                                            break  # Break out of inner loop
                                        if distance < min_distance:
                                            min_distance = distance
                                            assigned_cluster_id = (
                                                j + 1
                                            )  # Cluster ID starts from 1
                                    cluster_ids.append(assigned_cluster_id)
                                cluster_assignment_end_time = (
                                    time.time()
                                )  # Cluster assignment end time
                                cluster_assignment_elapsed_time = (
                                    cluster_assignment_end_time
                                    - cluster_assignment_start_time
                                )  # Cluster assignment elapsed time

                                # print(f"Cluster assignment completed in {cluster_assignment_elapsed_time:.3f} seconds.")

                                silhouette_avg = calculate_clustering_metrics_clear(
                                    states,
                                    cluster_ids,
                                    custom_distance.multi_distance,
                                    sample_size=100,
                                )
                                silhouette_avgs.append(silhouette_avg)

                                print(
                                    f"decay_factor: {decay_factor}, beta: {beta}, mu: {mu}, epsilon: {epsilon}"
                                )
                                print(
                                    f"Denstream Elapsed Time: {denstream_elapsed_time:.3f} seconds"
                                )
                                print(
                                    f"Cluster Assignment Elapsed Timr: {cluster_assignment_elapsed_time:.3f} seconds"
                                )
                                print(
                                    f"Total Elapsed Time: {denstream_elapsed_time + cluster_assignment_elapsed_time:.3f} seconds"
                                )
                                print(f"Number of core clusters: {len(core_clusters)}")
                                print(f"Silhouette Coefficient: {silhouette_avg:.3f}")
                                print("-" * 50)

                                # Use project output directory
                                output_file = (
                                    Path(get_multi_alg_output_path(sce_id))
                                    / "denstream_results.txt"
                                )
                                with open(output_file, "a", encoding="utf-8") as file:
                                    file.write(
                                        f"{decay_factor} & {beta} & {mu} & {epsilon} & {denstream_elapsed_time + cluster_assignment_elapsed_time:.3f} & {len(core_clusters)} & {silhouette_avg:.3f}\n"
                                    )

                            # avg_elapsed_time = sum(elapsed_times) / len(elapsed_times)
                            # avg_cluster_count = sum(cluster_counts) / len(cluster_counts)
                            # avg_silhouette_avg = sum(silhouette_avgs) / len(silhouette_avgs)
                            #
                            # print(f"decay_factor: {decay_factor}, beta: {beta}, mu: {mu}, epsilon: {epsilon}")
                            # print(f"Average Elapsed Time: {avg_elapsed_time:.3f} seconds")
                            # print(f"Average Number of Clusters: {avg_cluster_count:.2f}")
                            # print(f"Average Silhouette Coefficient: {avg_silhouette_avg:.3f}")
                            # print("-" * 50)


if __name__ == "__main__":
    main()
