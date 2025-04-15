import numpy as np
from scipy.interpolate import splprep, splev


class TrajectoryBuilder:
    """
    Class for building and smoothing drone trajectories.

    This class takes a set of drone positions and constructs a trajectory,
    optionally applying spline-based smoothing for better visualization.
    """

    def __init__(self, smooth=True, smooth_factor=0.5):
        """
        Initialize the trajectory builder.

        Args:
            smooth (bool): Whether to apply smoothing to the trajectory
            smooth_factor (float): Smoothing factor (0-1), higher values produce smoother trajectories
        """
        self.smooth = smooth
        self.smooth_factor = smooth_factor

    def build_trajectory(self, points, return_indices=False):
        """
        Build a trajectory from a list of points.

        This function takes a list of drone positions (which may include None values
        for positions that couldn't be determined) and creates a trajectory. If smoothing
        is enabled, it applies spline interpolation to create a smoother path.

        Args:
            points (list): List of points [(x1, y1), (x2, y2), ...] or None values
            return_indices (bool, optional): Whether to return original indices of valid points

        Returns:
            list: List of trajectory points, possibly smoothed
            list: (if return_indices=True) List of original indices of valid points
        """
        print(f"Building trajectory from {len(points)} points")
        print(f"Sample points: {points[:3] if points else []}")

        if len(points) < 2:
            if return_indices:
                return points, list(range(len(points)))
            return points

        # If smoothing is disabled
        if not self.smooth:
            if return_indices:
                return points, list(range(len(points)))
            return points

        # Remove None values from the list of points
        valid_points = []
        original_indices = []

        for i, point in enumerate(points):
            if point is not None:
                valid_points.append(point)
                original_indices.append(i)

        if len(valid_points) < 2:
            if return_indices:
                return valid_points, original_indices
            return valid_points

        # Split x and y coordinates
        x_coords = [p[0] for p in valid_points]
        y_coords = [p[1] for p in valid_points]

        try:
            # Parameterize by curve length
            tck, u = splprep([x_coords, y_coords], s=self.smooth_factor * len(valid_points),
                             k=min(3, len(valid_points) - 1))

            # Create a denser grid of points for the smoothed curve
            u_new = np.linspace(0, 1, num=len(valid_points) * 10)
            smoothed_points = splev(u_new, tck)

            # Convert back to a list of tuples
            smooth_trajectory = list(zip(smoothed_points[0], smoothed_points[1]))

            if return_indices:
                # Return both the smoothed trajectory and original indices
                return smooth_trajectory, original_indices
            return smooth_trajectory
        except Exception as e:
            print(f"Error smoothing trajectory: {e}")
            # In case of error, return the unsmoothed trajectory
            if return_indices:
                return valid_points, original_indices
            return valid_points