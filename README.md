# Space-Filling Minimum Energy Design

This repository implements the Minimum Energy Design (MED) method for generating space-filling designs in complex surfaces and multidimensional parameter spaces.

## Overview

The implementation focuses on generating points that maximize space coverage while respecting existing point distributions. The core functionality includes:

- `generate_med_points`: Creates new points using the Minimum Energy Design optimization technique
- `minimize_energy`: Finds points that minimize the energy function with respect to existing points
- `generate_sobol_med_points`: Uses Sobol sequences as candidates for generating MED points

## Method Attribution

This implementation is based on the Minimum Energy Design (MED) approach proposed by:

> Joseph, V. R., Tuo, R., Dasgupta, T., & Wu, C. F. J. (2015). Sequential Exploration of Complex Surfaces Using Minimum Energy Designs. Technometrics, 57(1), 64-74.

The method visualizes design points as charged particles in a box, minimizing the total potential energy to create optimal space-filling designs that can adapt to complex response surfaces.

## Usage

```python
# Example usage:
import numpy as np
from med_design import generate_med_points

# Create a dataset and define bounds
data = np.random.uniform(0, 15, (25, 2))  # existing points
bounds = [(0, 15) for _ in range(data.shape[1])]  # bounds for each dimension

# Generate new points using MED
X_new = generate_med_points(num_points=5, existing_points=data, 
                           bounds=bounds, num_restarts=25)