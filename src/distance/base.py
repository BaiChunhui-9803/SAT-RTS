import numpy as np
from scipy.optimize import linear_sum_assignment


class DistributionDistance:
    def __init__(self, state_1, state_2):
        self.state_1 = state_1
        self.state_2 = state_2

    def _extract_coordinates_and_health(self, state, army_type):
        """
        Extract all coordinates and health information for a specific army type from state.
        :param state: State dictionary
        :param army_type: Army type ('blue_army' or 'red_army')
        :return: 2D array containing all coordinates and 1D array containing health values
        """
        coordinates = []
        health_values = []
        for unit in state[army_type]:
            coordinates.append(unit[:2])  # Extract first two elements as coordinates
            health_values.append(unit[2])  # Assume health is the third element
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

        # Process blue_army
        coords1_blue, health1_blue = self._extract_coordinates_and_health(
            self.state_1["state"][0], "blue_army"
        )
        coords2_blue, health2_blue = self._extract_coordinates_and_health(
            self.state_2["state"][0], "blue_army"
        )

        # Process red_army
        coords1_red, health1_red = self._extract_coordinates_and_health(
            self.state_1["state"][0], "red_army"
        )
        coords2_red, health2_red = self._extract_coordinates_and_health(
            self.state_2["state"][0], "red_army"
        )

        # Define a helper function to calculate distance matrix and perform matching
        def calculate_army_distance_and_health_difference(
            coords1, coords2, health1, health2, max_distance
        ):
            if len(coords1) == 0 and len(coords2) == 0:
                return 0.0, 0.0  # If both arrays are empty, return (0.0, 0.0)
            elif len(coords1) == 0 or len(coords2) == 0:
                # If one is empty, return max distance and health difference multiplied by non-empty point count
                return max_distance * max(
                    len(coords1), len(coords2)
                ), 1.0  # Return 1.0 as health difference product
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

                # Calculate health difference for matched points
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

        # Calculate blue_army distance and health difference
        distance_blue, health_difference_blue = (
            calculate_army_distance_and_health_difference(
                coords1_blue, coords2_blue, health1_blue, health2_blue, max_distance
            )
        )
        total_distance += distance_blue
        total_health_difference += health_difference_blue

        # Calculate red_army distance and health difference
        distance_red, health_difference_red = (
            calculate_army_distance_and_health_difference(
                coords1_red, coords2_red, health1_red, health2_red, max_distance
            )
        )
        total_distance += distance_red
        total_health_difference += health_difference_red

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
        :return: tuple, (bool, float). First return value indicates whether two states are distributionally identical, second indicates health distribution difference
        """
        # Calculate distance and health difference between two states
        distance_calculator = DistributionDistance(obs1, obs2)
        distribution_distance, health_distance = (
            distance_calculator()
        )  # Call DistributionDistance's __call__ method to get distance value
        return distribution_distance, health_distance
