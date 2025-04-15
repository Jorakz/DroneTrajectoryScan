import cv2
import numpy as np
from DroneTrajectoryScan.src.feature_extraction import FeatureExtractor


class DroneLocalizer:
    """
    Class for determining the position of a drone on a global map.

    This class localizes drone images on a global map by extracting and matching
    features between the drone image and the map, then computing the homography
    transformation to find the drone's position.
    """

    def __init__(self, global_map, feature_method='sift'):
        """
        Initialize the drone localizer.

        Args:
            global_map (numpy.ndarray): Global map image
            feature_method (str): Feature extraction method to use ('sift', 'orb', 'akaze')
        """
        self.global_map = global_map
        self.extractor = FeatureExtractor(method=feature_method)

        # Pre-extract features from the global map for efficiency
        self.map_keypoints, self.map_descriptors = self.extractor.extract_features(global_map)
        print(f"Extracted {len(self.map_keypoints)} keypoints from the global map")

    def locate_drone_image(self, drone_image, min_matches=10):
        """
        Determine the position of a drone image on the global map.

        This function extracts features from the drone image, matches them with
        pre-extracted features from the global map, and computes a homography
        transformation to find the position of the drone image center on the map.

        Args:
            drone_image (numpy.ndarray): Drone image to locate
            min_matches (int): Minimum number of good matches required for reliable localization

        Returns:
            tuple: ((coords_x, coords_y), homography) - coordinates of the image center on the map
                  and the homography matrix, or None if localization fails
        """
        # Extract features from the drone image
        drone_keypoints, drone_descriptors = self.extractor.extract_features(drone_image)

        if drone_descriptors is None or len(drone_keypoints) < min_matches:
            print("Not enough keypoints in the drone image")
            return None

        # Match features
        matches = self.extractor.match_features(drone_descriptors, self.map_descriptors)

        if len(matches) < min_matches:
            print(f"Not enough matches: {len(matches)} < {min_matches}")
            return None

        # Get coordinates of matched points
        drone_pts = np.float32([drone_keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        map_pts = np.float32([self.map_keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Compute homography with RANSAC for robustness to outliers
        H, mask = cv2.findHomography(drone_pts, map_pts, cv2.RANSAC, 5.0)

        if H is None:
            print("Failed to compute homography")
            return None

        # Determine coordinates of the drone image center on the map
        h, w = drone_image.shape[:2]
        drone_center = np.array([[w / 2, h / 2]], dtype=np.float32).reshape(-1, 1, 2)
        map_center = cv2.perspectiveTransform(drone_center, H)

        return ((int(map_center[0][0][0]), int(map_center[0][0][1])), H)