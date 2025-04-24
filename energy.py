import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist


def energy(x_candidates, existing_points, k=4, epsilon=1e-8):
    """
    Objective function to minimize for finding the next point (Eq. 8).
    Calculates the sum of potential energy contributions between the candidate
    point(s) and all existing points.
    Args:
        x_candidates (np.array): The candidate point(s) (1D array for single point,
                                2D array [n_candidates x d] for multiple points).
        existing_points (np.array): Array of existing points (n x d).
        k (float): The exponent parameter.
    Returns:
        float or np.array: The calculated energy contribution sum(s).
    """

    x_candidates = np.asarray(x_candidates)

    # Handle single candidate case (convert to 2D)
    if x_candidates.ndim == 1:
        x_candidates = x_candidates.reshape(1, -1)
        single_point = True
    else:
        single_point = False

    if existing_points.shape[0] == 0:  # No points yet
        if single_point:
            return 0.0
        else:
            return np.zeros(x_candidates.shape[0])

    # Calculate distances between candidate points and existing points
    distances = cdist(x_candidates, existing_points)

    # Ensure distances are not zero
    distances = np.maximum(distances, epsilon)

    # Calculate the energy contribution for each candidate
    total_energies = np.sum(1.0 / (distances**k), axis=1)

    if single_point:
        return total_energies[0]
    else:
        return total_energies
