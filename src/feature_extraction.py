import cv2
import numpy as np

class FeatureExtractor:
    """Класс для извлечения и сопоставления ключевых точек."""

    def __init__(self, method='sift', matcher='flann'):
        """
        Инициализация экстрактора признаков.

        Args:
            method (str): Метод извлечения признаков ('sift', 'orb', 'akaze')
            matcher (str): Метод сопоставления ('flann', 'bf')
        """
        self.method = method
        self.matcher_type = matcher
        self._init_detector()
        self._init_matcher()

    def _init_detector(self):
        """Инициализация детектора ключевых точек."""
        if self.method == 'sift':
            self.detector = cv2.SIFT_create()
        elif self.method == 'orb':
            self.detector = cv2.ORB_create(nfeatures=10000)
        elif self.method == 'akaze':
            self.detector = cv2.AKAZE_create()
        else:
            raise ValueError(f"Неподдерживаемый метод извлечения признаков: {self.method}")

    def _init_matcher(self):
        """Инициализация сопоставителя ключевых точек."""
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
            raise ValueError(f"Неподдерживаемый тип сопоставителя: {self.matcher_type}")

    def extract_features(self, image):
        """
        Извлечение ключевых точек и дескрипторов из изображения.

        Args:
            image (numpy.ndarray): Входное изображение

        Returns:
            tuple: (ключевые точки, дескрипторы)
        """
        keypoints, descriptors = self.detector.detectAndCompute(image, None)
        return keypoints, descriptors

    def match_features(self, desc1, desc2, ratio_thresh=0.7):
        """
        Сопоставление дескрипторов с фильтрацией по отношению расстояний.

        Args:
            desc1, desc2 (numpy.ndarray): Дескрипторы для сопоставления
            ratio_thresh (float): Порог отношения для фильтрации сопоставлений

        Returns:
            list: Отфильтрованные сопоставления
        """
        matches = self.matcher.knnMatch(desc1, desc2, k=2)

        # Фильтрация сопоставлений по тесту отношения расстояний Лоу
        good_matches = []
        for m, n in matches:
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)

        return good_matches