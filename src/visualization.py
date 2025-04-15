import cv2
import numpy as np



class TrajectoryVisualizer:
    """
    Class for visualizing drone trajectories on a map.

    This class provides methods to visualize drone flight trajectories,
    individual drone positions, and the drone's field of view boxes on a global map.
    """

    def __init__(self, map_image, use_gpu=False):
        """
        Initialize the visualizer.

        Args:
            map_image (numpy.ndarray): Global map image
            use_gpu (bool): Whether to use GPU acceleration (if available)
        """
        self.map_image = map_image.copy()
        self.use_gpu = use_gpu

    def draw_trajectory(self, trajectory_points, color=(0, 0, 255), thickness=2):
        """
        Draw a trajectory on the map with numbered points.

        This method draws lines connecting consecutive trajectory points and
        adds numbered markers at each point. The first point is colored green,
        the last one red, and intermediate points blue.

        Args:
            trajectory_points (list): List of trajectory points
            color (tuple): Line color (B, G, R)
            thickness (int): Line thickness

        Returns:
            numpy.ndarray: Map image with the drawn trajectory
        """
        result = self.map_image.copy()

        if not trajectory_points:
            return result

        # Filter out None values
        valid_points_with_indices = [(i, p) for i, p in enumerate(trajectory_points) if p is not None]

        if not valid_points_with_indices:
            return result

        # Draw lines between consecutive points
        for i in range(1, len(valid_points_with_indices)):
            prev_idx, prev_point = valid_points_with_indices[i - 1]
            curr_idx, curr_point = valid_points_with_indices[i]

            pt1 = (int(prev_point[0]), int(prev_point[1]))
            pt2 = (int(curr_point[0]), int(curr_point[1]))
            cv2.line(result, pt1, pt2, color, thickness)

        # Draw all points with numbers
        for idx, (original_idx, point) in enumerate(valid_points_with_indices):
            x, y = int(point[0]), int(point[1])

            # Determine point color
            if idx == 0:
                point_color = (0, 255, 0)  # Green for the first point
            elif idx == len(valid_points_with_indices) - 1:
                point_color = (0, 0, 255)  # Red for the last point
            else:
                point_color = (255, 0, 0)  # Blue for intermediate points

            cv2.circle(result, (x, y), 8, point_color, -1)

            # Use original index + 1 for numbering (i.e., image number)
            text = str(original_idx + 1)

            # Add image number with black background for better visibility
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(text, font, 0.7, 2)[0]

            # Draw black rectangle for text background
            cv2.rectangle(result,
                          (x + 5, y - 5 - text_size[1]),
                          (x + 10 + text_size[0], y - 5),
                          (0, 0, 0), -1)

            # Draw white text
            cv2.putText(result, text, (x + 7, y - 7),
                        font, 0.7, (255, 255, 255), 2)

        return result

    def draw_labeled_trajectory(self, smooth_trajectory, original_points, original_indices,
                                line_color=(0, 0, 255), thickness=2):
        """
        Draw a trajectory with both smoothed lines and original numbered points.

        This method draws a smoothed trajectory line but places numbered markers
        at the original (unsmoothed) point positions for accurate reference.

        Args:
            smooth_trajectory (list): List of points in the smoothed trajectory
            original_points (list): List of original points with possible None values
            original_indices (list): List of indices of original points
            line_color (tuple): Line color (B, G, R)
            thickness (int): Line thickness

        Returns:
            numpy.ndarray: Map image with the drawn trajectory
        """
        result = self.map_image.copy()

        # Draw smoothed trajectory
        if smooth_trajectory and len(smooth_trajectory) > 1:
            for i in range(1, len(smooth_trajectory)):
                pt1 = (int(smooth_trajectory[i - 1][0]), int(smooth_trajectory[i - 1][1]))
                pt2 = (int(smooth_trajectory[i][0]), int(smooth_trajectory[i][1]))
                cv2.line(result, pt1, pt2, line_color, thickness)

        # Draw original points with their numbers
        for i, idx in enumerate(original_indices):
            if idx < len(original_points) and original_points[idx] is not None:
                point = original_points[idx]
                x, y = int(point[0]), int(point[1])

                # Choose point color
                if i == 0:
                    point_color = (0, 255, 0)  # Green for first point
                elif i == len(original_indices) - 1:
                    point_color = (0, 0, 255)  # Red for last point
                else:
                    point_color = (255, 0, 0)  # Blue for other points

                # Draw point
                cv2.circle(result, (x, y), 8, point_color, -1)

                # Add point number
                text = str(idx + 1)  # index + 1 to start numbering from 1
                font = cv2.FONT_HERSHEY_SIMPLEX
                text_size = cv2.getTextSize(text, font, 0.7, 2)[0]

                # Draw rectangle for text background
                cv2.rectangle(result,
                              (x + 5, y - 5 - text_size[1]),
                              (x + 10 + text_size[0], y - 5),
                              (0, 0, 0), -1)

                # Draw text
                cv2.putText(result, text, (x + 7, y - 7),
                            font, 0.7, (255, 255, 255), 2)

        return result

    def draw_drone_positions(self, positions, radius=5, color=(255, 0, 0)):
        """
        Draw drone positions on the map.

        This method places markers at each drone position and labels them
        with their corresponding image number.

        Args:
            positions (list): List of drone positions [(x1, y1), (x2, y2), ...]
            radius (int): Marker radius
            color (tuple): Marker color (B, G, R)

        Returns:
            numpy.ndarray: Map image with marked positions
        """
        result = self.map_image.copy()

        for i, pos in enumerate(positions):
            if pos is not None:
                cv2.circle(result, (int(pos[0]), int(pos[1])), radius, color, -1)
                # Add frame number
                cv2.putText(result, str(i + 1), (int(pos[0]) + radius, int(pos[1]) + radius),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return result

    def draw_drone_view_boxes(self, positions_with_homographies, drone_image_shape, color=(0, 255, 255)):
        """
        Draw rectangles showing the drone's field of view on the map.

        This method projects the corners of each drone image onto the map
        using the corresponding homography matrix to visualize what area
        the drone was seeing when it took each image.

        Args:
            positions_with_homographies (list): List of tuples (position, homography matrix)
            drone_image_shape (tuple): Drone image dimensions (height, width)
            color (tuple): Box color (B, G, R)

        Returns:
            numpy.ndarray: Map image with view boxes
        """
        result = self.map_image.copy()
        h, w = drone_image_shape

        for pos, H in positions_with_homographies:
            if pos is not None and H is not None:
                # Corner points of drone image
                drone_corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
                # Project corners onto map
                map_corners = cv2.perspectiveTransform(drone_corners, H).reshape(-1, 2)
                # Draw rectangle
                map_corners = np.int32(map_corners)
                cv2.polylines(result, [map_corners], True, color, 2)

        return result

    def create_trajectory_video(self, positions, output_path, fps=10, frame_size=None):
        """
        Create a video visualizing the drone's movement along the trajectory.

        This method generates a video that shows the drone moving along its
        trajectory, with the current position highlighted and previous positions
        marked with their corresponding image numbers.

        Args:
            positions (list): List of drone positions [(x1, y1), (x2, y2), ...]
            output_path (str): Path to save the video file
            fps (int): Frames per second
            frame_size (tuple): Video frame size (width, height)

        Returns:
            bool: True if video was created successfully, False otherwise
        """
        # Filter out None values
        valid_positions_with_indices = [(i, p) for i, p in enumerate(positions) if p is not None]
        if not valid_positions_with_indices:
            print("No valid positions for video creation")
            return False

        # Determine frame size
        if frame_size is None:
            h, w = self.map_image.shape[:2]
            frame_size = (w, h)

        # Create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

        if not video.isOpened():
            print(f"Failed to create VideoWriter for {output_path}")
            return False

        # Extract positions and their indices
        indices = [idx for idx, _ in valid_positions_with_indices]
        valid_positions = [pos for _, pos in valid_positions_with_indices]

        # Create sequence of frames
        for i in range(len(valid_positions)):
            # Create frame with trajectory up to current point
            current_trajectory = valid_positions[:i + 1]
            current_indices = indices[:i + 1]

            # Draw map with trajectory
            frame = self.map_image.copy()

            # Draw trajectory lines
            for j in range(1, len(current_trajectory)):
                pt1 = (int(current_trajectory[j - 1][0]), int(current_trajectory[j - 1][1]))
                pt2 = (int(current_trajectory[j][0]), int(current_trajectory[j][1]))
                cv2.line(frame, pt1, pt2, (0, 0, 255), 2)

            # Draw points with numbers
            for j, point in enumerate(current_trajectory):
                x, y = int(point[0]), int(point[1])

                # Current point - red, others - blue
                color = (0, 0, 255) if j == i else (255, 0, 0)
                cv2.circle(frame, (x, y), 8, color, -1)

                # Point number
                text = str(indices[j] + 1)
                font = cv2.FONT_HERSHEY_SIMPLEX
                text_size = cv2.getTextSize(text, font, 0.7, 2)[0]

                cv2.rectangle(frame,
                              (x + 5, y - 5 - text_size[1]),
                              (x + 10 + text_size[0], y - 5),
                              (0, 0, 0), -1)

                cv2.putText(frame, text, (x + 7, y - 7),
                            font, 0.7, (255, 255, 255), 2)

            # Add frame to video
            video.write(frame)

        # Repeat last frame for a delay at the end
        for _ in range(fps * 2):  # 2 seconds delay
            video.write(frame)

        video.release()
        print(f"Video saved to {output_path}")
        return True