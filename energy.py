import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist


def energy(x_candidate, existing_points, k=4, epsilon=1e-8):
    """
    Objective function to minimize for finding the next point (Eq. 8).
    Calculates the sum of potential energy contributions between the candidate
    point and all existing points.
    Args:
        x_candidate (np.array): The candidate point (1D array).
        existing_points (np.array): Array of existing points (n x d).
        k (float): The exponent parameter.
    Returns:
        float: The calculated energy contribution sum.
    """
    total_energy = 0.0

    if existing_points.shape[0] == 0:  # No points yet
        return 0.0

    # Calculate distances between candidate point and existing points
    distances = cdist(x_candidate.reshape(1, -1), existing_points)

    # Ensure distances are not zero
    distances = np.maximum(distances, epsilon)

    # Calculate the energy contribution for each distance
    total_energy = np.sum(1.0 / (distances**k))

    return total_energy
