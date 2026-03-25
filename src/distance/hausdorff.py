import numpy as np
from scipy.spatial.distance import cdist


class ModifiedHausdorffDistributionDistance:
    def __init__(self, state_1, state_2):
        self.state_1 = state_1
        self.state_2 = state_2

    def _extract_points(self, state, unit_type):
        """
        Extract point set coordinates for a specific unit type from state.
        Treat the point set as a discrete point cloud for Modified Hausdorff distance calculation.

        Modified Hausdorff distance is an improvement over standard Hausdorff distance,
        using average distance instead of maximum distance, making it more robust to outliers.

        :param state: State dictionary
        :param unit_type: Unit type ('self_units' or 'enemy_units')
        :return: numpy array representing point coordinates
        """
        if len(state[unit_type]) == 0:
            return np.array([[0, 0]])  # Empty point set, with a dummy point at origin

        # Extract all coordinates
        coordinates = []
        for unit in state[unit_type]:
            if unit["position"] is not None:
                x, y = unit["position"]
                coordinates.append([x, y])

        if len(coordinates) == 0:
            return np.array([[0, 0]])  # Empty point set

        return np.array(coordinates)

    def _calculate_modified_hausdorff_distance(self, points1, points2):
        """
        Calculate Modified Hausdorff distance between two point sets.

        Standard Hausdorff distance:
        HD(P1, P2) = max(h(P1, P2), h(P2, P1))
        where h(P1, P2) = max_{p∈P1} min_{q∈P2} ||p - q||

        Modified Hausdorff distance (more robust to outliers):
        MHD(P1, P2) = max(mean_{p∈P1} min_{q∈P2} ||p - q||,
                           mean_{q∈P2} min_{p∈P1} ||q - p||)

        Advantages:
        - Robust to outliers: uses mean instead of maximum
        - Considers global shape structure
        - Mathematical rigor

        :param points1: First point set (m×2)
        :param points2: Second point set (n×2)
        :return: Modified Hausdorff distance
        """
        # Handle empty point set cases
        if len(points1) == 0 and len(points2) == 0:
            return 0.0
        elif len(points1) == 0 or len(points2) == 0:
            return float(
                "inf"
            )  # Distance between empty and non-empty point set is infinity

        # If same point set
        if len(points1) == len(points2) and np.allclose(points1, points2):
            return 0.0

        # Use scipy to calculate distance matrix
        # dist_matrix[i, j] = ||points1[i] - points2[j]||
        dist_matrix = cdist(points1, points2, "euclidean")

        # Calculate bidirectional average nearest neighbor distance (core of Modified Hausdorff)
        # points1 -> points2: for each point in points1, find minimum distance to points2, then take average
        mean_dist_1_to_2 = np.mean(np.min(dist_matrix, axis=1))
        # points2 -> points1: for each point in points2, find minimum distance to points1, then take average
        mean_dist_2_to_1 = np.mean(np.min(dist_matrix, axis=0))

        # Modified Hausdorff distance: maximum of bidirectional average distances
        modified_hausdorff_dist = max(mean_dist_1_to_2, mean_dist_2_to_1)

        return modified_hausdorff_dist

    def _calculate_standard_hausdorff_distance(self, points1, points2):
        """
        Calculate standard Hausdorff distance as a comparison reference.

        Standard Hausdorff distance is very sensitive to outliers but useful in some cases.
        """
        if len(points1) == 0 and len(points2) == 0:
            return 0.0
        elif len(points1) == 0 or len(points2) == 0:
            return float("inf")

        if len(points1) == len(points2) and np.allclose(points1, points2):
            return 0.0

        dist_matrix = cdist(points1, points2, "euclidean")

        # Standard Hausdorff: use maximum
        max_dist_1_to_2 = np.max(np.min(dist_matrix, axis=1))
        max_dist_2_to_1 = np.max(np.min(dist_matrix, axis=0))

        standard_hausdorff_dist = max(max_dist_1_to_2, max_dist_2_to_1)

        return standard_hausdorff_dist

    def calculate_distance_and_distribution_difference(self):
        """
        Calculate Modified Hausdorff distance and distribution difference between two states.
        :return: tuple, (float, float). First return value is Modified Hausdorff distance, second is distribution weight difference
        """
        if self.state_1 == self.state_2:
            return 0.0, 0.0

        total_hausdorff_distance = 0.0
        total_weight_difference = 0.0

        # Process self_units (friendly units)
        points1_self = self._extract_points(self.state_1, "self_units")
        points2_self = self._extract_points(self.state_2, "self_units")

        # Process enemy_units (enemy units)
        points1_enemy = self._extract_points(self.state_1, "enemy_units")
        points2_enemy = self._extract_points(self.state_2, "enemy_units")

        # Calculate Modified Hausdorff distance for friendly units
        hausdorff_self = self._calculate_modified_hausdorff_distance(
            points1_self, points2_self
        )
        if hausdorff_self != float("inf"):
            total_hausdorff_distance += hausdorff_self

        # Calculate Modified Hausdorff distance for enemy units
        hausdorff_enemy = self._calculate_modified_hausdorff_distance(
            points1_enemy, points2_enemy
        )
        if hausdorff_enemy != float("inf"):
            total_hausdorff_distance += hausdorff_enemy

        # If any is infinity, return infinity
        if hausdorff_self == float("inf") or hausdorff_enemy == float("inf"):
            total_hausdorff_distance = float("inf")

        # Calculate weight difference (point count difference)
        # Modified Hausdorff distance mainly concerns shape structure, but we also calculate point count difference as supplementary information
        point_count_diff_self = abs(len(points1_self) - len(points2_self))
        point_count_diff_enemy = abs(len(points1_enemy) - len(points2_enemy))

        total_weight_difference = point_count_diff_self + point_count_diff_enemy

        return total_hausdorff_distance, total_weight_difference

    def __call__(self):
        return self.calculate_distance_and_distribution_difference()


class ModifiedHausdorffDistance:
    def __init__(self, threshold=0.5, grid_size=20.0):
        self.threshold = threshold
        self.grid_size = grid_size

    def multi_distance(self, obs1, obs2):
        """
        Use Modified Hausdorff distance to determine similarity of two states in point cloud distribution, and calculate distribution difference.
        :param obs1: First state
        :param obs2: Second state
        :return: tuple, (float, float). First return value is Modified Hausdorff distance, second is distribution weight difference
        """
        distance_calculator = ModifiedHausdorffDistributionDistance(obs1, obs2)
        hausdorff_distance, weight_difference = distance_calculator()
        return hausdorff_distance, weight_difference

    def calculate_batch_distances(self, states):
        """
        Calculate Modified Hausdorff distance matrix between batch states.
        :param states: List of states
        :return: Distance matrix and weight difference matrix, numpy arrays
        """
        n_states = len(states)
        distance_matrix = np.zeros((n_states, n_states))
        weight_difference_matrix = np.zeros((n_states, n_states))

        for i in range(n_states):
            for j in range(i, n_states):
                if i == j:
                    distance_matrix[i, j] = 0.0
                    weight_difference_matrix[i, j] = 0.0
                else:
                    dist, weight_diff = self.multi_distance(states[i], states[j])
                    distance_matrix[i, j] = dist
                    distance_matrix[j, i] = dist  # Symmetric matrix
                    weight_difference_matrix[i, j] = weight_diff
                    weight_difference_matrix[j, i] = weight_diff  # Symmetric matrix

        return distance_matrix, weight_difference_matrix

    def find_similar_states(self, target_state, states, threshold=None):
        """
        Find states similar to target state using Modified Hausdorff distance.
        :param target_state: Target state
        :param states: List of states
        :param threshold: Threshold, if None use the threshold from initialization
        :return: List of indices of similar states
        """
        if threshold is None:
            threshold = self.threshold

        similar_indices = []
        for i, state in enumerate(states):
            dist, _ = self.multi_distance(target_state, state)
            if dist <= threshold:
                similar_indices.append(i)

        return similar_indices

    def get_distance_name(self):
        """Return the name of the distance metric"""
        return "Modified Hausdorff Distance (Shape-based Point Cloud Metric)"
