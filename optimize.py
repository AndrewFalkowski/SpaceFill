import numpy as np
from scipy.optimize import minimize
from energy import energy
from utils import Normalizer


def minimize_energy(existing_points, bounds=None, num_restarts=10):
    """
    Find the point that minimizes energy with respect to existing points.

    This function performs multiple optimization attempts with different
    random starting points to find the global minimum of the energy function.

    Args:
        existing_points (np.array): Array of existing points (n x d).
        bounds (list of tuples, optional): List of (min, max) tuples for each dimension.
                                          Defaults to [(0, 1)] for each dimension.
        num_restarts (int, optional): Number of optimization attempts. Defaults to 10.

    Returns:
        np.array: The point that minimizes the energy function.
    """
    obj_func = lambda x: energy(x, existing_points)
    dim = existing_points.shape[1]

    # Set default bounds if not provided
    if bounds is None:
        bounds = [(0, 1) for _ in range(dim)]

    best_x = None
    best_result_val = float("inf")  # Initialize with infinity

    for restart in range(num_restarts):
        # Randomly generate initial guess within bounds
        x0 = np.random.uniform(bounds[0][0], bounds[0][1], size=dim)

        # Perform the optimization
        result = minimize(obj_func, x0, method="L-BFGS-B", bounds=bounds)

        if result.success and result.fun < best_result_val:
            best_result_val = result.fun
            best_x = result.x

    if best_x is None:
        print(f"Warning: Optimization failed, returning a random point.")
        best_x = np.random.uniform(bounds[0][0], bounds[0][1], size=dim)

    return best_x


def generate_med_points(num_points, existing_points, bounds=None, num_restarts=10):
    """
    Generate multiple points using Minimum Energy Design (MED) with domain bounds.

    This function sequentially adds points to the existing set by minimizing
    the energy function, creating a space-filling design within specified bounds.

    Args:
        num_points (int): Number of new points to generate.
        existing_points (np.array): Array of existing points (n x d).
        bounds (list of tuples, optional): List of (min, max) tuples for each dimension.
                                         If None, bounds will be determined from existing_points.
        num_restarts (int, optional): Number of optimization attempts per point.
                                     Defaults to 10.

    Returns:
        np.array: Array of newly generated points (num_points x d) within specified bounds.
    """
    # Create normalizer with specified bounds
    normalizer = Normalizer(bounds)

    # Scale existing points to [0,1] based on bounds
    scaled_existing_points = normalizer.fit_scale(existing_points)

    # Generate new points in [0,1] space
    new_scaled_points = np.zeros((num_points, existing_points.shape[1]))
    current_points = scaled_existing_points.copy()

    for i in range(num_points):
        # Use [0,1] bounds for optimization since we're in normalized space
        new_point = minimize_energy(current_points, num_restarts=num_restarts)
        new_scaled_points[i] = new_point
        current_points = np.vstack((current_points, new_point))

    # Unscale new points back to original bounds
    new_points = normalizer.unscale(new_scaled_points)

    return new_points


def generate_sobol_med_points(
    num_points, existing_points, bounds=None, num_candidates=000
):
    """
    Generate space-filling points using Sobol sequences and minimum energy design.

    This function generates Sobol sequence points as candidates and selects
    those with minimum energy contribution to create a space-filling design.

    Args:
        num_points (int): Number of new points to generate.
        existing_points (np.array): Array of existing points (n x d).
        bounds (list of tuples, optional): List of (min, max) tuples for each dimension.
                                         If None, bounds will be determined from existing_points.
        num_candidates (int, optional): Number of Sobol candidate points to generate for each selection.
                                       Defaults to 1000.

    Returns:
        np.array: Array of newly generated points (num_points x d) within specified bounds.
    """
    # Import Sobol sequence generator (Ensure scipy is installed)
    try:
        from scipy.stats import qmc
    except ImportError:
        raise ImportError(
            "SciPy 1.7.0 or later is required for Sobol sequence generation. "
            "Please install with 'pip install scipy>=1.7.0'"
        )

    # Create normalizer with specified bounds
    normalizer = Normalizer(bounds)

    # Scale existing points to [0,1] based on bounds
    scaled_existing_points = normalizer.fit_scale(existing_points)

    # Dimensions of the problem
    n_dim = existing_points.shape[1]

    # Generate new points in [0,1] space
    new_scaled_points = np.zeros((num_points, n_dim))
    current_points = scaled_existing_points.copy()

    # Generate Sobol sequence candidates
    sampler = qmc.Sobol(d=n_dim, scramble=True)
    candidates = sampler.random(num_candidates)

    for i in range(num_points):

        # Calculate energy for each candidate
        energies = np.zeros(num_candidates)
        for j in range(num_candidates):
            energies[j] = energy(candidates[j], current_points)

        # Select the candidate with minimum energy
        best_idx = np.argmin(energies)
        best_point = candidates[best_idx]

        # Store the new point
        new_scaled_points[i] = best_point

        # Update the current points for the next iteration
        current_points = np.vstack((current_points, best_point))

    # Unscale new points back to original bounds
    new_points = normalizer.unscale(new_scaled_points)

    return new_points
