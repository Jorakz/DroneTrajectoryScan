import numpy as np
from scipy.interpolate import splprep, splev


class TrajectoryBuilder:
    """Класс для построения и сглаживания траектории дрона."""

    def __init__(self, smooth=True, smooth_factor=0.5):
        """
        Инициализация построителя траектории.

        Args:
            smooth (bool): Применять ли сглаживание
            smooth_factor (float): Степень сглаживания (0-1)
        """
        self.smooth = smooth
        self.smooth_factor = smooth_factor

    def build_trajectory(self, points):
        """
        Построение траектории по точкам.

        Args:
            points (list): Список точек [(x1, y1), (x2, y2), ...]

        Returns:
            list: Список точек траектории, возможно сглаженной
        """
        print(f"Построена траектория из {len(points)} точек")
        print(f"Примеры точек траектории: {points[:3] if points else []}")
        if len(points) < 2:
            return points

        if not self.smooth:
            return points

        # Удаление None-значений из списка точек
        valid_points = [p for p in points if p is not None]
        if len(valid_points) < 2:
            return valid_points

        # Разделение x и y координат
        x_coords = [p[0] for p in valid_points]
        y_coords = [p[1] for p in valid_points]

        # Параметризация по длине кривой
        tck, u = splprep([x_coords, y_coords], s=self.smooth_factor * len(valid_points),
                         k=min(3, len(valid_points) - 1))

        # Создание более плотной сетки точек для сглаженной кривой
        u_new = np.linspace(0, 1, num=len(valid_points) * 10)
        smoothed_points = splev(u_new, tck)

        # Преобразование обратно в список кортежей
        return list(zip(smoothed_points[0], smoothed_points[1]))