import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import wasserstein_distance

# Try to import POT library, provide warning if unavailable
try:
    import ot

    POT_AVAILABLE = True
    print("POT library imported successfully, using true optimal transport algorithm")
except ImportError:
    POT_AVAILABLE = False
    print(
        "Warning: POT library not installed. Please run 'pip install pot' to install POT library for true Wasserstein distance calculation"
    )
    print("Currently using simplified implementation as fallback")


class WassersteinDistributionDistance:
    def __init__(self, state_1, state_2):
        self.state_1 = state_1
        self.state_2 = state_2

    def _extract_points_as_distribution(self, state, unit_type):
        """
        Extract point set for a specific unit type from state and treat it as an empirical probability distribution.
        Directly use point set positions, each point assigned uniform weight 1/n.

        Transform point set A = {a₁,…,a_m} to discrete measure μ = Σ_{i=1}^{m} (1/m) δ_{a_i}

        :param state: State dictionary
        :param unit_type: Unit type ('self_units' or 'enemy_units')
        :return: Two numpy arrays representing point coordinates and corresponding uniform weights
        """
        if len(state[unit_type]) == 0:
            return np.array([[0, 0]]), np.array(
                [1.0]
            )  # Empty distribution, with a dummy point at origin

        # Extract all coordinates
        coordinates = []
        for unit in state[unit_type]:
            if unit["position"] is not None:
                # Normalize coordinates to [0, 1] range
                x, y = unit["position"]
                norm_x = x
                norm_y = y
                coordinates.append([norm_x, norm_y])

        if len(coordinates) == 0:
            return np.array([[0, 0]]), np.array([1.0])  # Empty distribution

        points = np.array(coordinates)
        n_points = len(points)

        # Each point is assigned uniform weight 1/n
        uniform_weights = np.ones(n_points) / n_points

        return points, uniform_weights

    def _calculate_wasserstein_distance(
        self, points1, weights1, points2, weights2, p=1
    ):
        """
        Calculate Wasserstein distance (Earth Mover's Distance) between two discrete probability distributions.

        For discrete measures μ = Σ_{i=1}^{m} w_i δ_{a_i} and ν = Σ_{j=1}^{n} v_j δ_{b_j},
        Wₚ(μ,ν) = ( min_{T∈ℝ^{m×n}} Σ_{i,j} T_{ij}·‖a_i − b_j‖^p )^{1/p}

        Constraints: T_{ij} ≥ 0,  Σ_j T_{ij} = w_i,  Σ_i T_{ij} = v_j

        Use POT library to solve true optimal transport problem.

        :param points1: Point coordinates of first distribution (m×2)
        :param weights1: Weights of first distribution (m,)
        :param points2: Point coordinates of second distribution (n×2)
        :param weights2: Weights of second distribution (n,)
        :param p: Order of Wasserstein distance, default is 1
        :return: Wasserstein distance
        """
        # Handle empty distribution cases
        if len(points1) == 0 or len(points2) == 0:
            return float(
                "inf"
            )  # Distance between empty and non-empty distribution is infinity

        # If same distribution
        if (
            len(points1) == len(points2)
            and np.allclose(points1, points2)
            and np.allclose(weights1, weights2)
        ):
            return 0.0

        # Prefer using POT library for true optimal transport calculation
        if POT_AVAILABLE:
            try:
                return self._calculate_wasserstein_distance_pot(
                    points1, weights1, points2, weights2, p
                )
            except Exception as e:
                print(f"POT calculation failed, using fallback method: {e}")
                return self._calculate_wasserstein_distance_fallback(
                    points1, weights1, points2, weights2, p
                )
        else:
            return self._calculate_wasserstein_distance_fallback(
                points1, weights1, points2, weights2, p
            )

    def _calculate_wasserstein_distance_pot(
        self, points1, weights1, points2, weights2, p=1
    ):
        """
        Calculate true Wasserstein distance using POT library
        """
        # Ensure weights are normalized (POT requires probability distributions to sum to 1)
        weights1_normalized = weights1 / np.sum(weights1)
        weights2_normalized = weights2 / np.sum(weights2)

        # Calculate cost matrix: C_{ij} = ||a_i - b_j||^p
        # Use vectorized computation for efficiency
        if p == 1:
            # For p=1, use Euclidean distance directly
            cost_matrix = np.linalg.norm(points1[:, np.newaxis] - points2, axis=2)
        else:
            # For p>1, use p-order distance
            cost_matrix = np.linalg.norm(points1[:, np.newaxis] - points2, axis=2) ** p

        # Use POT library's Earth Mover's Distance algorithm
        # ot.emd() solves the optimal transport problem, returns optimal transport plan
        optimal_transport_plan = ot.emd(
            weights1_normalized, weights2_normalized, cost_matrix
        )

        # Calculate Wasserstein distance: Σ_{i,j} T_{ij} * C_{ij}
        wasserstein_distance = np.sum(optimal_transport_plan * cost_matrix)

        # For p>1 case, need to take p-th root
        if p != 1:
            wasserstein_distance = wasserstein_distance ** (1.0 / p)

        return wasserstein_distance

    def _calculate_wasserstein_distance_fallback(
        self, points1, weights1, points2, weights2, p=1
    ):
        """
        Fallback method: simplified implementation when POT library is unavailable
        Note: This is not true Wasserstein distance, but serves as a backup solution
        """
        n1, n2 = len(points1), len(points2)

        # Calculate cost matrix: C_{ij} = ||a_i - b_j||^p
        cost_matrix = np.zeros((n1, n2))
        for i in range(n1):
            for j in range(n2):
                cost_matrix[i, j] = np.linalg.norm(points1[i] - points2[j]) ** p

        # Create augmented matrix to handle weight imbalance (simplified method based on assignment problem)
        max_len = max(n1, n2)
        augmented_matrix = np.full((max_len, max_len), np.max(cost_matrix))
        augmented_matrix[:n1, :n2] = cost_matrix

        # Use Hungarian algorithm to solve assignment problem
        row_ind, col_ind = linear_sum_assignment(augmented_matrix)

        # Calculate total transport cost (simplified version)
        total_cost = 0.0
        for r, c in zip(row_ind, col_ind):
            if r < n1 and c < n2:
                # Transport between real points
                total_cost += augmented_matrix[r, c] * min(weights1[r], weights2[c])
            elif r < n1:
                # Point from first distribution transported to dummy point
                total_cost += augmented_matrix[r, c] * weights1[r]
            elif c < n2:
                # Dummy point transported to point in second distribution
                total_cost += augmented_matrix[r, c] * weights2[c]

        # Normalize cost
        wasserstein_distance = total_cost / max_len

        # For p>1 case, need to take p-th root
        if p != 1:
            wasserstein_distance = wasserstein_distance ** (1.0 / p)

        return wasserstein_distance

    def calculate_distance_and_distribution_difference(self):
        """
        Calculate Wasserstein distance and distribution difference between two states.
        Uses the method of directly treating point sets as empirical probability distributions.
        :return: tuple, (float, float). First return value is Wasserstein distance, second is distribution weight difference
        """
        if self.state_1 == self.state_2:
            return 0.0, 0.0

        total_wasserstein_distance = 0.0
        total_weight_difference = 0.0

        # Process self_units (friendly units)
        points1_self, weights1_self = self._extract_points_as_distribution(
            self.state_1, "self_units"
        )
        points2_self, weights2_self = self._extract_points_as_distribution(
            self.state_2, "self_units"
        )

        # Process enemy_units (enemy units)
        points1_enemy, weights1_enemy = self._extract_points_as_distribution(
            self.state_1, "enemy_units"
        )
        points2_enemy, weights2_enemy = self._extract_points_as_distribution(
            self.state_2, "enemy_units"
        )

        # Calculate Wasserstein distance for friendly units (using p=1)
        wasserstein_self = self._calculate_wasserstein_distance(
            points1_self, weights1_self, points2_self, weights2_self, p=2
        )
        if wasserstein_self != float("inf"):
            total_wasserstein_distance += wasserstein_self

        # Calculate Wasserstein distance for enemy units (using p=1)
        wasserstein_enemy = self._calculate_wasserstein_distance(
            points1_enemy, weights1_enemy, points2_enemy, weights2_enemy, p=2
        )
        if wasserstein_enemy != float("inf"):
            total_wasserstein_distance += wasserstein_enemy

        # If any is infinity, return infinity
        if wasserstein_self == float("inf") or wasserstein_enemy == float("inf"):
            total_wasserstein_distance = float("inf")

        # Calculate weight difference (distribution mass difference)
        # Here we calculate the difference in total distribution mass as weight difference
        # Friendly unit mass difference
        mass1_self = np.sum(weights1_self)
        mass2_self = np.sum(weights2_self)
        weight_diff_self = abs(mass1_self - mass2_self)

        # Enemy unit mass difference
        mass1_enemy = np.sum(weights1_enemy)
        mass2_enemy = np.sum(weights2_enemy)
        weight_diff_enemy = abs(mass1_enemy - mass2_enemy)

        total_weight_difference = weight_diff_self + weight_diff_enemy

        return total_wasserstein_distance, total_weight_difference

    def __call__(self):
        return self.calculate_distance_and_distribution_difference()


class WassersteinDistance:
    def __init__(self, threshold=0.5, grid_size=20.0):
        self.threshold = threshold
        self.grid_size = grid_size

    def multi_distance(self, obs1, obs2):
        """
        Use Wasserstein distance to determine whether two states are distributionally identical, and calculate distribution difference.
        :param obs1: First state
        :param obs2: Second state
        :return: tuple, (float, float). First return value is Wasserstein distance, second is distribution weight difference
        """
        distance_calculator = WassersteinDistributionDistance(obs1, obs2)
        wasserstein_distance, weight_difference = distance_calculator()
        return wasserstein_distance, weight_difference

    def calculate_batch_distances(self, states):
        """
        Calculate Wasserstein distance matrix between batch states.
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
        Find states similar to target state using Wasserstein distance.
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
        return "Wasserstein Distance (Earth Mover's Distance)"
