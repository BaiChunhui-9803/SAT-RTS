import numpy as np
from scipy.optimize import linear_sum_assignment


class DistributionDistance:
    def __init__(self, state_1, state_2):
        self.state_1 = state_1
        self.state_2 = state_2

    def _extract_coordinates_and_health(self, state, unit_type):
        """
        Extract all coordinates and health information for a specific unit type from state.
        :param state: State dictionary
        :param unit_type: Unit type ('self_units' or 'enemy_units')
        :return: 2D array containing all coordinates and 1D array containing health values
        """
        coordinates = []
        health_values = []

        for unit in state[unit_type]:
            if unit["position"] is not None:
                coordinates.append(unit["position"])  # Extract coordinates (x, y)
                # Since new structure has no health value, we default all unit health values to 1.0
                health_values.append(1.0)  # Default health value

        return np.array(coordinates), np.array(health_values)

    def calculate_distance_and_health_difference(self):
        """
        Calculate the distance and health distribution difference between two states.
        :return: tuple, (float, float). First return value is coordinate distribution distance, second is health distribution difference
        """
        if self.state_1 == self.state_2:
            return 0.0, 0.0

        total_distance = 0.0
        total_health_difference = 0.0

        # Define a large value as the distance to dummy points
        max_distance = 1.0

        # Process self_units (friendly units - red)
        coords1_self, health1_self = self._extract_coordinates_and_health(
            self.state_1, "self_units"
        )
        coords2_self, health2_self = self._extract_coordinates_and_health(
            self.state_2, "self_units"
        )

        # Process enemy_units (enemy units - blue)
        coords1_enemy, health1_enemy = self._extract_coordinates_and_health(
            self.state_1, "enemy_units"
        )
        coords2_enemy, health2_enemy = self._extract_coordinates_and_health(
            self.state_2, "enemy_units"
        )

        # Define a helper function to calculate distance matrix and perform matching
        def calculate_army_distance_and_health_difference(
            coords1, coords2, health1, health2, max_distance
        ):
            if len(coords1) == 0 and len(coords2) == 0:
                return 0.0, 0.0  # If both arrays are empty, return (0.0, 0.0)
            elif len(coords1) == 0 or len(coords2) == 0:
                # If one is empty, return max distance and health difference multiplied by non-empty point count
                return max_distance * max(len(coords1), len(coords2)), max(
                    len(coords1), len(coords2)
                )  # Return point count difference
            else:
                # Calculate coordinate distance matrix
                distance_matrix = np.linalg.norm(
                    coords1[:, np.newaxis] - coords2, axis=2
                )
                max_len = max(len(coords1), len(coords2))
                distance_matrix = np.pad(
                    distance_matrix,
                    ((0, max_len - len(coords1)), (0, max_len - len(coords2))),
                    mode="constant",
                    constant_values=max_distance,
                )
                # Match nearest points
                row_ind, col_ind = linear_sum_assignment(distance_matrix)
                # Calculate coordinate distance for matched points
                total_distance = distance_matrix[row_ind, col_ind].sum()

                # Calculate health difference for matched points (since all are 1.0, this mainly calculates count difference)
                health_difference_product = 0.0
                for r, c in zip(row_ind, col_ind):
                    if r < len(health1) and c < len(health2):
                        health_difference_product += abs(health1[r] - health2[c])
                    elif r < len(health1):
                        health_difference_product += abs(
                            health1[r]
                        )  # Difference with dummy point
                    elif c < len(health2):
                        health_difference_product += abs(
                            health2[c]
                        )  # Difference with dummy point

                return total_distance, health_difference_product

        # Calculate distance and health difference for self_units
        distance_self, health_difference_self = (
            calculate_army_distance_and_health_difference(
                coords1_self, coords2_self, health1_self, health2_self, max_distance
            )
        )
        total_distance += distance_self
        total_health_difference += health_difference_self

        # Calculate distance and health difference for enemy_units
        distance_enemy, health_difference_enemy = (
            calculate_army_distance_and_health_difference(
                coords1_enemy, coords2_enemy, health1_enemy, health2_enemy, max_distance
            )
        )
        total_distance += distance_enemy
        total_health_difference += health_difference_enemy

        return total_distance, total_health_difference

    def __call__(self):
        return self.calculate_distance_and_health_difference()


class CustomDistance:
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def multi_distance(self, obs1, obs2):
        """
        Determine whether two states are distributionally identical and calculate health distribution difference.
        :param obs1: First state
        :param obs2: Second state
        :return: tuple, (float, float). First return value is distribution distance, second is health distribution difference
        """
        # Calculate distance and health difference between two states
        distance_calculator = DistributionDistance(obs1, obs2)
        distribution_distance, health_distance = (
            distance_calculator()
        )  # Call DistributionDistance's __call__ method to get distance value
        return distribution_distance, health_distance

    def calculate_batch_distances(self, states):
        """
        Calculate distance matrix between batch states.
        :param states: List of states
        :return: Distance matrix, numpy array
        """
        n_states = len(states)
        distance_matrix = np.zeros((n_states, n_states))
        health_difference_matrix = np.zeros((n_states, n_states))

        for i in range(n_states):
            for j in range(i, n_states):
                if i == j:
                    distance_matrix[i, j] = 0.0
                    health_difference_matrix[i, j] = 0.0
                else:
                    dist, health_diff = self.multi_distance(states[i], states[j])
                    distance_matrix[i, j] = dist
                    distance_matrix[j, i] = dist  # Symmetric matrix
                    health_difference_matrix[i, j] = health_diff
                    health_difference_matrix[j, i] = health_diff  # Symmetric matrix

        return distance_matrix, health_difference_matrix

    def find_similar_states(self, target_state, states, threshold=None):
        """
        Find states similar to target state.
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
