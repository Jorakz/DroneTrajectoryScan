import cv2



class FeatureExtractor:
    """
    Class for extracting and matching keypoints in images.

    This class provides functionality to detect keypoints in images and match them
    between different images using various algorithms like SIFT, ORB, and AKAZE.
    """

    def __init__(self, method='sift', matcher='flann'):
        """
        Initialize the feature extractor.

        Args:
            method (str): Feature extraction method ('sift', 'orb', 'akaze')
            matcher (str): Matching method ('flann', 'bf')
        """
        self.method = method
        self.matcher_type = matcher
        self._init_detector()
        self._init_matcher()

    def _init_detector(self):
        """
        Initialize the keypoint detector based on the selected method.

        Creates the appropriate detector object (SIFT) and
        assigns it to self.detector.
        """
        if self.method == 'sift':
            self.detector = cv2.SIFT_create()
        else:
            raise ValueError(f"Unsupported feature extraction method: {self.method}")

    def _init_matcher(self):
        """
        Initialize the feature matcher based on the selected method.

        Creates the appropriate matcher object (FLANN or BFMatcher) with
        parameters suitable for the selected feature detection method.
        """
        if self.matcher_type == 'flann':
            if self.method == 'orb':
                FLANN_INDEX_LSH = 6
                index_params = dict(algorithm=FLANN_INDEX_LSH,
                                    table_number=6,
                                    key_size=12,
                                    multi_probe_level=1)
            else:
                FLANN_INDEX_KDTREE = 1
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)

            search_params = dict(checks=50)
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        elif self.matcher_type == 'bf':
            if self.method == 'sift':
                self.matcher = cv2.BFMatcher(cv2.NORM_L2)
            else:
                self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        else:
            raise ValueError(f"Unsupported matcher type: {self.matcher_type}")

    def extract_features(self, image):
        """
        Extract keypoints and descriptors from an image.

        Args:
            image (numpy.ndarray): Input image

        Returns:
            tuple: (keypoints, descriptors) where keypoints is a list of cv2.KeyPoint objects
                  and descriptors is a numpy array of feature descriptors
        """
        keypoints, descriptors = self.detector.detectAndCompute(image, None)
        return keypoints, descriptors

    def match_features(self, desc1, desc2, ratio_thresh=0.7):
        """
        Match descriptors between two images using ratio test filtering.

        Args:
            desc1, desc2 (numpy.ndarray): Descriptors to match
            ratio_thresh (float): Ratio threshold for Lowe's ratio test filtering

        Returns:
            list: Filtered good matches that pass the ratio test
        """
        matches = self.matcher.knnMatch(desc1, desc2, k=2)

        # Filter matches using Lowe's ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)

        return good_matches