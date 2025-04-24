import numpy as np


import numpy as np


class Normalizer:
    """
    A class for transforming data to and from unit scale (0-1) based on specified bounds.

    This transformer handles scaling data to the unit interval and
    transforming it back to the original scale using provided domain bounds.

    Attributes:
        min_vals (np.array): Minimum bound values for each dimension
        range_vals (np.array): Range (max-min) values for each dimension
        is_fitted (bool): Whether the transformer has been fitted
    """

    def __init__(self, bounds=None):
        """
        Initialize the Normalizer.

        Args:
            bounds (list of tuples, optional): List of (min, max) tuples for each dimension.
                                              If None, bounds will be determined from data.
        """
        self.min_vals = None
        self.range_vals = None
        self.is_fitted = False
        self.bounds = bounds

    def fit(self, data):
        """
        Compute scaling parameters based on data or specified bounds.

        Args:
            data (np.array): Input data array of shape (n_samples, n_dimensions)

        Returns:
            self: The fitted normalizer
        """
        n_dimensions = data.shape[1]

        if self.bounds is None:
            # Use data min/max if no bounds are specified
            self.min_vals = np.min(data, axis=0)
            max_vals = np.max(data, axis=0)
        else:
            # Use specified bounds
            if len(self.bounds) != n_dimensions:
                raise ValueError(
                    f"Expected {n_dimensions} bounds but got {len(self.bounds)}"
                )

            self.min_vals = np.array([bound[0] for bound in self.bounds])
            max_vals = np.array([bound[1] for bound in self.bounds])

        self.range_vals = max_vals - self.min_vals

        # Handle zero ranges to prevent division by zero
        self.range_vals = np.where(self.range_vals == 0, 1, self.range_vals)

        self.is_fitted = True
        return self

    def scale(self, data):
        """
        Transform data to unit scale (0-1) based on fitted bounds.

        Args:
            data (np.array): Input data array to scale

        Returns:
            np.array: Scaled data in range [0, 1] for each dimension

        Raises:
            ValueError: If the normalizer has not been fitted
        """
        if not self.is_fitted:
            raise ValueError("Normalizer must be fitted before scaling data")

        # Scale the data
        return (data - self.min_vals) / self.range_vals

    def fit_scale(self, data):
        """
        Fit to data/bounds and transform data to unit scale in one step.

        Args:
            data (np.array): Input data array of shape (n_samples, n_dimensions)

        Returns:
            np.array: Scaled data in range [0, 1] for each dimension
        """
        self.fit(data)
        return self.scale(data)

    def unscale(self, scaled_data):
        """
        Transform data from unit scale back to original scale.

        Args:
            scaled_data (np.array): Data in unit scale [0, 1]

        Returns:
            np.array: Data transformed back to the original scale

        Raises:
            ValueError: If the normalizer has not been fitted
        """
        if not self.is_fitted:
            raise ValueError("Normalizer must be fitted before unscaling data")

        # Transform back to original scale
        return (scaled_data * self.range_vals) + self.min_vals


def euclidean_distance_sq(p1, p2):
    """Calculate squared Euclidean distance."""
    return np.sum((p1 - p2) ** 2)


def euclidean_distance(p1, p2):
    """Calculate Euclidean distance."""
    return np.sqrt(np.sum((p1 - p2) ** 2))
