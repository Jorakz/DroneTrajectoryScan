import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm
from src.feature_extraction import FeatureExtractor
from src.localization import DroneLocalizer
from src.trajectory import TrajectoryBuilder
from src.visualization import TrajectoryVisualizer
import config


def parse_args():
    parser = argparse.ArgumentParser(description='Восстановление траектории дрона по изображениям')
    parser.add_argument('--map', default='data/global_map.png',
                        help='Путь к глобальной карте')
    parser.add_argument('--images_dir', default='data/crops',
                        help='Директория с изображениями дрона')
    parser.add_argument('--output_dir', default='data/output',
                        help='Директория для сохранения результатов')
    parser.add_argument('--method', default='sift', choices=['sift', 'orb', 'akaze'],
                        help='Метод извлечения ключевых точек')
    parser.add_argument('--smooth', action='store_true',
                        help='Применять сглаживание траектории')
    parser.add_argument('--visualize_positions', action='store_true',
                        help='Визуализировать позиции дрона на карте')
    return parser.parse_args()


def main():
    args = parse_args()

    # Создание выходной директории, если она не существует
    os.makedirs(args.output_dir, exist_ok=True)

    # Загрузка глобальной карты
    global_map = cv2.imread(args.map)
    if global_map is None:
        raise FileNotFoundError(f"Не удалось загрузить карту: {args.map}")

    print(f"Размер глобальной карты: {global_map.shape}")

    # Инициализация локализатора дрона
    localizer = DroneLocalizer(global_map, feature_method=args.method)

    # Получение списка изображений дрона
    drone_images = sorted(
        [f for f in os.listdir(args.images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))],
        key=lambda name: int(name.split('_')[1].split('.')[0])
    )

    if not drone_images:
        raise FileNotFoundError(f"Не найдены изображения в директории: {args.images_dir}")

    # Локализация каждого изображения дрона
    positions = []
    homographies = []

    print("Локализация изображений дрона...")
    for img_name in tqdm(drone_images):
        img_path = os.path.join(args.images_dir, img_name)
        drone_img = cv2.imread(img_path)

        if drone_img is None:
            print(f"Пропуск {img_name}: не удалось загрузить изображение")
            positions.append(None)
            homographies.append(None)
            continue

        # Определение позиции дрона на карте

        position_and_homography = localizer.locate_drone_image(drone_img)
        if position_and_homography:
            position, homography = position_and_homography
            positions.append(position)
            homographies.append(homography)
        else:
            print(f"Не удалось локализовать изображение: {img_name}")
            positions.append(None)
            homographies.append(None)

    # Построение траектории
    trajectory_builder = TrajectoryBuilder(smooth=args.smooth)
    trajectory = trajectory_builder.build_trajectory(positions)

    # Визуализация результатов
    visualizer = TrajectoryVisualizer(global_map)

    # Рисование траектории
    trajectory_image = visualizer.draw_trajectory(trajectory)
    cv2.imwrite(os.path.join(args.output_dir, "trajectory.jpg"), trajectory_image)

    # Опционально: визуализация позиций
    if args.visualize_positions:
        positions_image = visualizer.draw_drone_positions(positions)
        cv2.imwrite(os.path.join(args.output_dir, "positions.jpg"), positions_image)

    print(f"Результаты сохранены в директории: {args.output_dir}")


if __name__ == "__main__":
    main()