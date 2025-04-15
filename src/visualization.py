import cv2
import numpy as np
import os


class TrajectoryVisualizer:
    """Класс для визуализации траектории дрона на карте."""

    def __init__(self, map_image, use_gpu=False):
        """
        Инициализация визуализатора.

        Args:
            map_image (numpy.ndarray): Изображение глобальной карты
            use_gpu (bool): Использовать ли GPU для ускорения (если доступно)
        """
        self.map_image = map_image.copy()
        self.use_gpu = use_gpu

    def draw_trajectory(self, trajectory_points, color=(0, 0, 255), thickness=2):
        """
        Отрисовка траектории на карте с нумерацией точек.

        Args:
            trajectory_points (list): Список точек траектории
            color (tuple): Цвет линии (B, G, R)
            thickness (int): Толщина линии

        Returns:
            numpy.ndarray: Изображение карты с нарисованной траекторией
        """
        result = self.map_image.copy()

        if not trajectory_points:
            return result

        # Фильтрация None-значений
        valid_points_with_indices = [(i, p) for i, p in enumerate(trajectory_points) if p is not None]

        if not valid_points_with_indices:
            return result

        # Отрисовка линий между последовательными точками
        for i in range(1, len(valid_points_with_indices)):
            prev_idx, prev_point = valid_points_with_indices[i - 1]
            curr_idx, curr_point = valid_points_with_indices[i]

            pt1 = (int(prev_point[0]), int(prev_point[1]))
            pt2 = (int(curr_point[0]), int(curr_point[1]))
            cv2.line(result, pt1, pt2, color, thickness)

        # Отрисовка всех точек с номерами
        for idx, (original_idx, point) in enumerate(valid_points_with_indices):
            x, y = int(point[0]), int(point[1])

            # Определяем цвет точки
            if idx == 0:
                point_color = (0, 255, 0)  # Зеленый для первой точки
            elif idx == len(valid_points_with_indices) - 1:
                point_color = (0, 0, 255)  # Красный для последней точки
            else:
                point_color = (255, 0, 0)  # Синий для промежуточных точек

            cv2.circle(result, (x, y), 8, point_color, -1)

            # Используем оригинальный индекс + 1 для нумерации (т.е. номер изображения)
            text = str(original_idx + 1)

            # Добавляем номер изображения с черным фоном для лучшей видимости
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(text, font, 0.7, 2)[0]

            # Рисуем черный прямоугольник для фона текста
            cv2.rectangle(result,
                          (x + 5, y - 5 - text_size[1]),
                          (x + 10 + text_size[0], y - 5),
                          (0, 0, 0), -1)

            # Рисуем белый текст
            cv2.putText(result, text, (x + 7, y - 7),
                        font, 0.7, (255, 255, 255), 2)

        return result

    def draw_drone_positions(self, positions, radius=5, color=(255, 0, 0)):
        """
        Отрисовка позиций дрона на карте.

        Args:
            positions (list): Список позиций дрона [(x1, y1), (x2, y2), ...]
            radius (int): Радиус маркера
            color (tuple): Цвет маркера (B, G, R)

        Returns:
            numpy.ndarray: Изображение карты с отмеченными позициями
        """
        result = self.map_image.copy()

        for i, pos in enumerate(positions):
            if pos is not None:
                cv2.circle(result, (int(pos[0]), int(pos[1])), radius, color, -1)
                # Добавление номера кадра
                cv2.putText(result, str(i + 1), (int(pos[0]) + radius, int(pos[1]) + radius),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return result

    def draw_drone_view_boxes(self, positions_with_homographies, drone_image_shape, color=(0, 255, 255)):
        """
        Отрисовка прямоугольников, показывающих область обзора дрона на карте.

        Args:
            positions_with_homographies (list): Список кортежей (позиция, матрица гомографии)
            drone_image_shape (tuple): Размеры изображения дрона (высота, ширина)
            color (tuple): Цвет рамки (B, G, R)

        Returns:
            numpy.ndarray: Изображение карты с областями обзора
        """
        result = self.map_image.copy()
        h, w = drone_image_shape

        for pos, H in positions_with_homographies:
            if pos is not None and H is not None:
                # Угловые точки изображения дрона
                drone_corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
                # Проекция углов на карту
                map_corners = cv2.perspectiveTransform(drone_corners, H).reshape(-1, 2)
                # Отрисовка прямоугольника
                map_corners = np.int32(map_corners)
                cv2.polylines(result, [map_corners], True, color, 2)

        return result

    def create_trajectory_video(self, positions, output_path, fps=10, frame_size=None):
        """
        Создание видео с визуализацией движения дрона по траектории.

        Args:
            positions (list): Список позиций дрона [(x1, y1), (x2, y2), ...]
            output_path (str): Путь для сохранения видео
            fps (int): Частота кадров видео
            frame_size (tuple): Размер кадра видео (ширина, высота), если None - используется размер карты

        Returns:
            bool: True если видео создано успешно, False иначе
        """
        # Фильтрация None значений
        valid_positions_with_indices = [(i, p) for i, p in enumerate(positions) if p is not None]
        if not valid_positions_with_indices:
            print("Нет валидных позиций для создания видео")
            return False

        # Определение размера кадра
        if frame_size is None:
            h, w = self.map_image.shape[:2]
            frame_size = (w, h)

        # Создание объекта VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

        if not video.isOpened():
            print(f"Не удалось создать объект VideoWriter для {output_path}")
            return False

        # Извлечение позиций и их индексов
        indices = [idx for idx, _ in valid_positions_with_indices]
        valid_positions = [pos for _, pos in valid_positions_with_indices]

        # Создание последовательности кадров
        for i in range(len(valid_positions)):
            # Создаем кадр с траекторией до текущей точки
            current_trajectory = valid_positions[:i + 1]
            current_indices = indices[:i + 1]

            # Рисуем карту с траекторией
            frame = self.map_image.copy()

            # Рисуем пройденную траекторию
            for j in range(1, len(current_trajectory)):
                pt1 = (int(current_trajectory[j - 1][0]), int(current_trajectory[j - 1][1]))
                pt2 = (int(current_trajectory[j][0]), int(current_trajectory[j][1]))
                cv2.line(frame, pt1, pt2, (0, 0, 255), 2)

            # Рисуем точки с номерами
            for j, point in enumerate(current_trajectory):
                x, y = int(point[0]), int(point[1])

                # Текущая точка - красная, остальные - синие
                color = (0, 0, 255) if j == i else (255, 0, 0)
                cv2.circle(frame, (x, y), 8, color, -1)

                # Номер точки
                text = str(indices[j] + 1)
                font = cv2.FONT_HERSHEY_SIMPLEX
                text_size = cv2.getTextSize(text, font, 0.7, 2)[0]

                cv2.rectangle(frame,
                              (x + 5, y - 5 - text_size[1]),
                              (x + 10 + text_size[0], y - 5),
                              (0, 0, 0), -1)

                cv2.putText(frame, text, (x + 7, y - 7),
                            font, 0.7, (255, 255, 255), 2)

            # Добавляем кадр в видео
            video.write(frame)

        # Повторяем последний кадр несколько раз для задержки в конце
        for _ in range(fps * 2):  # 2 секунды задержки
            video.write(frame)

        video.release()
        print(f"Видео сохранено в {output_path}")
        return True