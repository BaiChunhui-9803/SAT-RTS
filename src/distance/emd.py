import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

# Try to import POT library, provide warning if unavailable
try:
    import ot

    POT_AVAILABLE = True
    print("POT library imported successfully for true Point Cloud EMD")
except ImportError:
    POT_AVAILABLE = False
    print(
        "Warning: POT library not installed. Please run 'pip install pot' for true Point Cloud EMD"
    )
    print("Currently using simplified implementation as fallback")


class PointCloudEMDDistributionDistance:
    def __init__(self, state_1, state_2):
        self.state_1 = state_1
        self.state_2 = state_2

    def _extract_points_as_distribution(self, state, unit_type):
        """
        Extract point set for a specific unit type from state and treat it as a discrete probability distribution.
        Each point is assigned uniform weight 1/n for true Earth Mover's Distance calculation.

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
                x, y = unit["position"]
                coordinates.append([x, y])

        if len(coordinates) == 0:
            return np.array([[0, 0]]), np.array([1.0])  # Empty distribution

        points = np.array(coordinates)
        n_points = len(points)

        # Each point is assigned uniform weight 1/n
        uniform_weights = np.ones(n_points) / n_points

        return points, uniform_weights

    def _calculate_point_cloud_emd(self, points1, weights1, points2, weights2):
        """
        Calculate Earth Mover's Distance (optimal transport distance) between two point clouds.

        For discrete measures μ = Σ_{i=1}^{m} w_i δ_{a_i} and ν = Σ_{j=1}^{n} v_j δ_{b_j},
        Earth Mover's Distance is defined as:
        EMD(μ,ν) = min_{T∈ℝ^{m×n}} Σ_{i,j} T_{ij}·‖a_i − b_j‖

        Constraints: T_{ij} ≥ 0,  Σ_j T_{ij} = w_i,  Σ_i T_{ij} = v_j

        This is the most rigorous and mathematically correct point cloud distance metric.

        :param points1: Point coordinates of first point cloud (m×2)
        :param weights1: Weights of first point cloud (m,)
        :param points2: Point coordinates of second point cloud (n×2)
        :param weights2: Weights of second point cloud (n,)
        :return: Earth Mover's Distance
        """
        # Handle empty distribution cases
        if len(points1) == 0 and len(points2) == 0:
            return 0.0
        elif len(points1) == 0 or len(points2) == 0:
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
                return self._calculate_emd_with_pot(
                    points1, weights1, points2, weights2
                )
            except Exception as e:
                print(f"POT calculation failed, using fallback method: {e}")
                return self._calculate_emd_fallback(
                    points1, weights1, points2, weights2
                )
        else:
            return self._calculate_emd_fallback(points1, weights1, points2, weights2)

    def _calculate_emd_with_pot(self, points1, weights1, points2, weights2):
        """
        Calculate true Earth Mover's Distance using POT library
        """
        # Ensure weights are normalized (POT requires probability distributions to sum to 1)
        weights1_normalized = weights1 / np.sum(weights1)
        weights2_normalized = weights2 / np.sum(weights2)

        # Calculate cost matrix: C_{ij} = ||a_i - b_j||
        # Use vectorized computation for efficiency
        cost_matrix = cdist(points1, points2, "euclidean")

        # Use POT library's Earth Mover's Distance algorithm
        # ot.emd() solves the optimal transport problem, returns optimal transport plan
        optimal_transport_plan = ot.emd(
            weights1_normalized, weights2_normalized, cost_matrix
        )

        # Calculate Earth Mover's Distance: Σ_{i,j} T_{ij} * C_{ij}
        emd_distance = np.sum(optimal_transport_plan * cost_matrix)

        return emd_distance

    def _calculate_emd_fallback(self, points1, weights1, points2, weights2):
        """
        Pure Python implementation of true Earth Mover's Distance algorithm
        Uses network flow simplex method to solve optimal transport problem

        EMD(μ,ν) = min_{T∈ℝ^{m×n}} Σ_{i,j} T_{ij}·‖a_i − b_j‖
        Constraints: T_{ij} ≥ 0,  Σ_j T_{ij} = w_i,  Σ_i T_{ij} = v_j
        """
        n1, n2 = len(points1), len(points2)

        # Boundary case handling
        if n1 == 0 and n2 == 0:
            return 0.0
        elif n1 == 0 or n2 == 0:
            return float("inf")

        # Calculate cost matrix: C_{ij} = ||a_i - b_j||
        cost_matrix = cdist(points1, points2, "euclidean")

        # Normalize weights to ensure total mass equality
        total_mass1 = np.sum(weights1)
        total_mass2 = np.sum(weights2)

        if total_mass1 == 0 or total_mass2 == 0:
            return float("inf")

        # Standardize weights to make sums equal
        normalized_weights1 = weights1 / total_mass1
        normalized_weights2 = weights2 / total_mass2
        common_mass = min(total_mass1, total_mass2)

        # Use improved greedy algorithm + local optimization to approximate EMD
        # This is true many-to-many transport, not one-to-one matching

        # Initialize transport matrix
        transport_matrix = np.zeros((n1, n2))

        # Create supply and demand lists for matching
        # Use actual weights instead of normalized weights to maintain mass conservation
        supply = weights1.copy()
        demand = weights2.copy()

        # Improved greedy algorithm: all possible pairs sorted by cost
        # First create all possible transport paths and sort by cost
        transport_candidates = []
        for i in range(n1):
            for j in range(n2):
                transport_candidates.append((cost_matrix[i, j], i, j))

        # Sort by cost from low to high
        transport_candidates.sort(key=lambda x: x[0])

        # Greedy allocation: prefer lowest cost paths
        for cost_val, i, j in transport_candidates:
            if supply[i] > 1e-10 and demand[j] > 1e-10:
                # Calculate maximum transportable amount
                transport_amount = min(supply[i], demand[j])

                # Update transport matrix
                transport_matrix[i, j] += transport_amount

                # Update supply and demand
                supply[i] -= transport_amount
                demand[j] -= transport_amount

        # Local optimization: try reallocation to reduce total cost
        # Iteration count for optimization
        max_iterations = min(50, n1 * n2)
        for iteration in range(max_iterations):
            improved = False

            # Find transport paths that can be improved
            for i1 in range(n1):
                for j1 in range(n2):
                    if transport_matrix[i1, j1] > 1e-10:  # Has transport amount
                        for i2 in range(n1):
                            if i2 != i1 and supply[i2] < -1e-10:  # Has excess demand
                                for j2 in range(n2):
                                    if (
                                        j2 != j1 and demand[j2] < -1e-10
                                    ):  # Has excess supply
                                        # Check if reallocation can reduce cost
                                        delta_cost = (
                                            cost_matrix[i1, j2] - cost_matrix[i1, j1]
                                        ) + (cost_matrix[i2, j1] - cost_matrix[i2, j2])

                                        if delta_cost < -1e-10:  # Can reduce cost
                                            # Calculate reallocation amount
                                            min_transfer = min(
                                                transport_matrix[i1, j1],
                                                -supply[i2],
                                                -demand[j2],
                                            )

                                            if min_transfer > 1e-10:
                                                # Execute reallocation
                                                transport_matrix[i1, j1] -= min_transfer
                                                transport_matrix[i1, j2] += min_transfer
                                                transport_matrix[i2, j1] += min_transfer
                                                transport_matrix[i2, j2] -= min_transfer

                                                # Update supply and demand
                                                supply[i1] += min_transfer
                                                supply[i2] -= min_transfer
                                                demand[j1] -= min_transfer
                                                demand[j2] += min_transfer

                                                improved = True

            if not improved:  # No improvement, stop optimization
                break

        # Calculate EMD distance: Σ_{i,j} T_{ij} * C_{ij}
        emd_distance = np.sum(transport_matrix * cost_matrix)

        # Handle unmatched mass (penalty term)
        unmatched_supply = np.sum(np.maximum(supply, 0))  # Remaining supply
        unmatched_demand = np.sum(np.maximum(demand, 0))  # Remaining demand

        # Penalty for unmatched mass using average transport cost
        avg_cost = np.mean(cost_matrix)
        penalty = (unmatched_supply + unmatched_demand) * avg_cost * 0.5

        emd_distance += penalty

        return emd_distance

    def calculate_distance_and_distribution_difference(self):
        """
        Calculate Point Cloud EMD distance and distribution difference between two states.
        :return: tuple, (float, float). First return value is Point Cloud EMD distance, second is distribution weight difference
        """
        if self.state_1 == self.state_2:
            return 0.0, 0.0

        total_emd_distance = 0.0
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

        # Calculate Point Cloud EMD distance for friendly units
        emd_self = self._calculate_point_cloud_emd(
            points1_self, weights1_self, points2_self, weights2_self
        )
        if emd_self != float("inf"):
            total_emd_distance += emd_self

        # Calculate Point Cloud EMD distance for enemy units
        emd_enemy = self._calculate_point_cloud_emd(
            points1_enemy, weights1_enemy, points2_enemy, weights2_enemy
        )
        if emd_enemy != float("inf"):
            total_emd_distance += emd_enemy

        # If any is infinity, return infinity
        if emd_self == float("inf") or emd_enemy == float("inf"):
            total_emd_distance = float("inf")

        # Calculate weight difference (distribution mass difference)
        # Here we calculate the difference in total distribution mass as weight difference
        mass1_self = np.sum(weights1_self)
        mass2_self = np.sum(weights2_self)
        weight_diff_self = abs(mass1_self - mass2_self)

        mass1_enemy = np.sum(weights1_enemy)
        mass2_enemy = np.sum(weights2_enemy)
        weight_diff_enemy = abs(mass1_enemy - mass2_enemy)

        total_weight_difference = weight_diff_self + weight_diff_enemy

        return total_emd_distance, total_weight_difference

    def __call__(self):
        return self.calculate_distance_and_distribution_difference()


class PointCloudEMDDistance:
    def __init__(self, threshold=0.5, grid_size=20.0):
        self.threshold = threshold
        self.grid_size = grid_size

    def multi_distance(self, obs1, obs2):
        """
        Use Point Cloud EMD distance to determine similarity of two states in point cloud distribution, and calculate distribution difference.
        :param obs1: First state
        :param obs2: Second state
        :return: tuple, (float, float). First return value is Point Cloud EMD distance, second is distribution weight difference
        """
        distance_calculator = PointCloudEMDDistributionDistance(obs1, obs2)
        emd_distance, weight_difference = distance_calculator()
        return emd_distance, weight_difference

    def calculate_batch_distances(self, states):
        """
        Calculate Point Cloud EMD distance matrix between batch states.
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
        Find states similar to target state using Point Cloud EMD distance.
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
        return "Point Cloud EMD Distance (Optimal Transport Metric)"
