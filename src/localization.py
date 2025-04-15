import cv2
import numpy as np
from DroneTrajectoryScan.src.feature_extraction import FeatureExtractor


class DroneLocalizer:
    """Класс для определения положения дрона на глобальной карте."""

    def __init__(self, global_map, feature_method='sift'):
        """
        Инициализация локализатора дрона.

        Args:
            global_map (numpy.ndarray): Глобальная карта
            feature_method (str): Метод извлечения признаков
        """
        self.global_map = global_map
        self.extractor = FeatureExtractor(method=feature_method)

        # Предварительное извлечение признаков из глобальной карты
        self.map_keypoints, self.map_descriptors = self.extractor.extract_features(global_map)
        print(f"Извлечено {len(self.map_keypoints)} ключевых точек из глобальной карты")

    def locate_drone_image(self, drone_image, min_matches=10):
        """
        Определение положения изображения дрона на глобальной карте.

        Args:
            drone_image (numpy.ndarray): Изображение с дрона
            min_matches (int): Минимальное количество хороших сопоставлений

        Returns:
            tuple: (coords_x, coords_y) - координаты центра изображения на карте,
                  или None в случае неудачи
        """
        # Извлечение признаков из изображения дрона
        drone_keypoints, drone_descriptors = self.extractor.extract_features(drone_image)

        if drone_descriptors is None or len(drone_keypoints) < min_matches:
            print("Недостаточно ключевых точек на изображении дрона")
            return None

        # Сопоставление признаков
        matches = self.extractor.match_features(drone_descriptors, self.map_descriptors)

        if len(matches) < min_matches:
            print(f"Недостаточно сопоставлений: {len(matches)} < {min_matches}")
            return None

        # Получение координат сопоставленных точек
        drone_pts = np.float32([drone_keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        map_pts = np.float32([self.map_keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Вычисление гомографии с RANSAC для устойчивости к выбросам
        H, mask = cv2.findHomography(drone_pts, map_pts, cv2.RANSAC, 5.0)

        if H is None:
            print("Не удалось вычислить гомографию")
            return None

        # Определение координат центра изображения дрона на карте
        h, w = drone_image.shape[:2]
        drone_center = np.array([[w / 2, h / 2]], dtype=np.float32).reshape(-1, 1, 2)
        map_center = cv2.perspectiveTransform(drone_center, H)

        return ((int(map_center[0][0][0]), int(map_center[0][0][1])), H)