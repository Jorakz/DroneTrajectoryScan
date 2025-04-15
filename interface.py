import gradio as gr
import os
import cv2
import numpy as np
from PIL import Image
from src.localization import DroneLocalizer
import config

# === Глобальные переменные ===
map_file = None
map_image = None
selected_dir = None
image_files = []
current_index = 0
localizer = None
positions = []
homographies = []

# === Загрузка карты ===
def load_map(path):
    global map_file, map_image, localizer
    map_file = path.name
    map_image = cv2.imread(map_file)
    if map_image is None:
        return None
    localizer = DroneLocalizer(map_image, feature_method="sift")
    return Image.fromarray(cv2.cvtColor(map_image, cv2.COLOR_BGR2RGB))

# === Выбор папки с кадрами ===
def select_folder(folder):
    global selected_dir, image_files, current_index, positions, homographies

    if not os.path.isdir(folder):
        return None, "Папка не найдена", None

    selected_dir = folder
    current_index = 0
    image_files = sorted(
        [f for f in os.listdir(folder) if f.endswith(('.png', '.jpg'))],
        key=lambda x: int(x.split('_')[1].split('.')[0])
    )
    positions = [None] * len(image_files)
    homographies = [None] * len(image_files)

    if image_files:
        return show_crop_and_box(current_index)
    return None, "Нет изображений в папке", None

# === Отрисовка кадра + прямоугольника на карте ===
def draw_selected_box_on_trajectory(index):
    global localizer, map_image, positions, homographies, image_files

    # Проверка: загружена ли карта
    if map_image is None:
        print("Глобальная карта не загружена. Сначала загрузите карту.")
        return None

    # Проверка: инициализирован ли локализатор
    if localizer is None:
        print("Локализатор не инициализирован.")
        return None

    # Попробуем загрузить trajectory.jpg
    trajectory_path = "data/output/trajectory.jpg"
    if os.path.exists(trajectory_path):
        trajectory = cv2.imread(trajectory_path)
    else:
        trajectory = map_image.copy()  # безопасно, т.к. проверили выше

    filename = os.path.join(selected_dir, image_files[index])
    drone_img = cv2.imread(filename)
    if drone_img is None:
        print(f"Не удалось загрузить изображение дрона: {filename}")
        return Image.fromarray(cv2.cvtColor(trajectory, cv2.COLOR_BGR2RGB))

    # Если позиция и гомография ещё не посчитаны — считаем
    if positions[index] is None or homographies[index] is None:
        result = localizer.locate_drone_image(drone_img)
        if result:
            positions[index], homographies[index] = result
        else:
            print(f"Не удалось локализовать изображение: {filename}")
            return Image.fromarray(cv2.cvtColor(trajectory, cv2.COLOR_BGR2RGB))

    # Рисуем прямоугольник обзора дрона
    h, w = drone_img.shape[:2]
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(corners, homographies[index])
    projected = np.int32(projected)
    cv2.polylines(trajectory, [projected], True, (0, 255, 255), 3)

    return Image.fromarray(cv2.cvtColor(trajectory, cv2.COLOR_BGR2RGB))


# === Показ текущего кадра + карта ===
def show_crop_and_box(index):
    path = os.path.join(selected_dir, image_files[index])
    crop = Image.open(path)
    map_with_box = draw_selected_box_on_trajectory(index)
    return crop, image_files[index], map_with_box

# === Переключение кадров ===
def next_crop():
    global current_index
    if image_files:
        current_index = (current_index + 1) % len(image_files)
        return show_crop_and_box(current_index)
    return None, "", None

def prev_crop():
    global current_index
    if image_files:
        current_index = (current_index - 1) % len(image_files)
        return show_crop_and_box(current_index)
    return None, "", None

# === Запуск main.py (траектория) ===
def run_main():
    if map_file and selected_dir:
        os.system(f"python main.py --map \"{map_file}\" --images_dir \"{selected_dir}\" --output_dir \"data/output\"")
        return Image.open("data/output/trajectory.jpg")
    return None

# === Очистка ===
def clear_all():
    global map_file, map_image, selected_dir, image_files, current_index, localizer, positions, homographies
    map_file = None
    map_image = None
    selected_dir = None
    image_files = []
    current_index = 0
    localizer = None
    positions = []
    homographies = []
    return None, None, "", None

# === Интерфейс Gradio ===
with gr.Blocks() as demo:
    gr.Markdown("# Drone Navigation Gradio Interface")

    with gr.Row():
        map_output = gr.Image(label="Map / Trajectory")
    with gr.Row():
        run_btn = gr.Button("Build Trajectory")
        clear_btn = gr.Button("Clear")

    with gr.Row():
        map_input = gr.File(label="Upload Map Image (.jpg or .png)", file_types=[".jpg", ".png"])
        map_input.change(load_map, inputs=map_input, outputs=map_output)

        folder_input = gr.Textbox(label="Crop Folder (relative to 'data/crops')")
        crop_output = gr.Image(label="Preview Crop")
        current_name = gr.Textbox(label="Current Filename", interactive=False)

        folder_input.change(fn=select_folder, inputs=folder_input, outputs=[crop_output, current_name, map_output])

        with gr.Row():
            next_btn = gr.Button("Next Crop")
            prev_btn = gr.Button("Previous Crop")

            prev_btn.click(fn=prev_crop, outputs=[crop_output, current_name, map_output])
            next_btn.click(fn=next_crop, outputs=[crop_output, current_name, map_output])

    run_btn.click(fn=run_main, outputs=map_output)
    clear_btn.click(fn=clear_all, outputs=[map_output, crop_output, folder_input, map_output])

demo.launch()
