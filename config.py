# Параметры извлечения признаков
FEATURE_EXTRACTION = {
    'sift': {
        'nfeatures': 0,  # 0 означает извлечение всех возможных признаков
        'matcher': 'flann',
        'ratio_thresh': 0.7,
    },
    'orb': {
        'nfeatures': 10000,
        'matcher': 'bf',
        'ratio_thresh': 0.75,
    },
    'akaze': {
        'matcher': 'bf',
        'ratio_thresh': 0.7,
    }
}

# Параметры локализации
LOCALIZATION = {
    'min_matches': 10,  # Минимальное количество хороших сопоставлений
    'ransac_threshold': 5.0,  # Порог RANSAC для вычисления гомографии
}

# Параметры траектории
TRAJECTORY = {
    'smooth': True,  # Применять ли сглаживание траектории
    'smooth_factor': 0.5,  # Степень сглаживания (0-1)
}

# Параметры визуализации
VISUALIZATION = {
    'trajectory_color': (0, 0, 255),  # Красный (BGR)
    'trajectory_thickness': 2,
    'start_point_color': (0, 255, 0),  # Зеленый (BGR)
    'end_point_color': (0, 0, 255),  # Красный (BGR)
    'position_color': (255, 0, 0),  # Синий (BGR)
    'position_radius': 5,
    'view_box_color': (0, 255, 255),  # Желтый (BGR)
}