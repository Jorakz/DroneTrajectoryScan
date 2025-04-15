# Configuration parameters for drone trajectory reconstruction

# Feature extraction parameters
# Controls how keypoints are detected and matched between images
FEATURE_EXTRACTION = {
    'sift': {
        'nfeatures': 0,  # 0 means extract all possible features
        'matcher': 'flann',  # FLANN-based matcher for efficiency
        'ratio_thresh': 0.7,  # Lowe's ratio test threshold for filtering matches
    },
    'orb': {
        'nfeatures': 10000,  # Maximum number of features to detect
        'matcher': 'bf',  # Brute-force matcher
        'ratio_thresh': 0.75,  # Higher threshold for ORB features
    },
    'akaze': {
        'matcher': 'bf',  # Brute-force matcher
        'ratio_thresh': 0.7,  # Ratio test threshold
    }
}

# Localization parameters
# Controls how drone images are localized on the global map
LOCALIZATION = {
    'min_matches': 10,  # Minimum number of good matches required for reliable localization
    'ransac_threshold': 5.0,  # RANSAC threshold for computing homography
}

# Trajectory parameters
# Controls how the drone trajectory is constructed
TRAJECTORY = {
    'smooth': True,  # Whether to apply smoothing to the trajectory
    'smooth_factor': 0.5,  # Smoothing factor (0-1), higher values produce smoother trajectories
}

# Visualization parameters
# Controls the appearance of visualized trajectories and points
VISUALIZATION = {
    'trajectory_color': (0, 0, 255),  # Red in BGR format
    'trajectory_thickness': 2,  # Line thickness
    'start_point_color': (0, 255, 0),  # Green in BGR format
    'end_point_color': (0, 0, 255),  # Red in BGR format
    'position_color': (255, 0, 0),  # Blue in BGR format
    'position_radius': 5,  # Radius of position markers
    'view_box_color': (0, 255, 255),  # Yellow in BGR format
}