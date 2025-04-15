import gradio as gr
import os
import cv2
import numpy as np
from PIL import Image
from src.localization import DroneLocalizer


# Global variables
map_file = None  # Path to the global map file
map_image = None  # Loaded global map image
selected_dir = None  # Selected directory with drone images
image_files = []  # List of drone image files
current_index = 0  # Index of the currently displayed image
localizer = None  # DroneLocalizer instance
positions = []  # List of detected drone positions
homographies = []  # List of homography matrices


def load_map(path):
    """
    Load a map image and initialize the drone localizer.

    This function loads the selected map image, initializes the DroneLocalizer
    with it, and returns the image for display in the interface.

    Args:
        path: File path to the map image

    Returns:
        PIL.Image: Loaded map image for display, or None if loading fails
    """
    global map_file, map_image, localizer
    map_file = path.name
    map_image = cv2.imread(map_file)
    if map_image is None:
        return None
    localizer = DroneLocalizer(map_image, feature_method="sift")
    return Image.fromarray(cv2.cvtColor(map_image, cv2.COLOR_BGR2RGB))


def select_folder(folder):
    """
    Select a folder containing drone images.

    This function loads the list of image files from the selected folder,
    initializes arrays for positions and homographies, and displays the first image.

    Args:
        folder: Path to the folder containing drone images

    Returns:
        tuple: (crop_image, filename, map_with_box) for the first image,
               or error message if the folder is invalid
    """
    global selected_dir, image_files, current_index, positions, homographies

    if not os.path.isdir(folder):
        return None, "Folder not found", None

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
    return None, "No images in folder", None


def draw_selected_box_on_trajectory(index):
    """
    Draw a bounding box for the selected drone image on the trajectory/map.

    This function localizes the drone image on the map (if not already done),
    and draws a yellow rectangle showing the drone's field of view on the map.

    Args:
        index: Index of the drone image to display

    Returns:
        PIL.Image: Map image with the drone's field of view rectangle
    """
    global localizer, map_image, positions, homographies, image_files

    # Check if map is loaded
    if map_image is None:
        print("Global map not loaded. Please load a map first.")
        return None

    # Check if localizer is initialized
    if localizer is None:
        print("Localizer not initialized.")
        return None

    # Try to load trajectory.jpg
    trajectory_path = "data/output/trajectory.jpg"
    if os.path.exists(trajectory_path):
        trajectory = cv2.imread(trajectory_path)
    else:
        trajectory = map_image.copy()  # Safe since we checked above

    filename = os.path.join(selected_dir, image_files[index])
    drone_img = cv2.imread(filename)
    if drone_img is None:
        print(f"Failed to load drone image: {filename}")
        return Image.fromarray(cv2.cvtColor(trajectory, cv2.COLOR_BGR2RGB))

    # If position and homography not yet calculated - calculate them
    if positions[index] is None or homographies[index] is None:
        result = localizer.locate_drone_image(drone_img)
        if result:
            positions[index], homographies[index] = result
        else:
            print(f"Failed to localize image: {filename}")
            return Image.fromarray(cv2.cvtColor(trajectory, cv2.COLOR_BGR2RGB))

    # Draw drone's field of view rectangle
    h, w = drone_img.shape[:2]
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(corners, homographies[index])
    projected = np.int32(projected)
    cv2.polylines(trajectory, [projected], True, (0, 255, 255), 3)

    return Image.fromarray(cv2.cvtColor(trajectory, cv2.COLOR_BGR2RGB))


def show_crop_and_box(index):
    """
    Display the current drone image and its position on the map.

    Args:
        index: Index of the image to display

    Returns:
        tuple: (crop_image, filename, map_with_box) for display in the interface
    """
    path = os.path.join(selected_dir, image_files[index])
    crop = Image.open(path)
    map_with_box = draw_selected_box_on_trajectory(index)
    return crop, image_files[index], map_with_box


def next_crop():
    """
    Show the next drone image in the sequence.

    Returns:
        tuple: (crop_image, filename, map_with_box) for the next image
    """
    global current_index
    if image_files:
        current_index = (current_index + 1) % len(image_files)
        return show_crop_and_box(current_index)
    return None, "", None


def prev_crop():
    """
    Show the previous drone image in the sequence.

    Returns:
        tuple: (crop_image, filename, map_with_box) for the previous image
    """
    global current_index
    if image_files:
        current_index = (current_index - 1) % len(image_files)
        return show_crop_and_box(current_index)
    return None, "", None


def run_main():
    """
    Run the main trajectory reconstruction process.

    Executes the main.py script with the currently selected map and image folder,
    then loads and returns the resulting trajectory image.

    Returns:
        PIL.Image: The generated trajectory image
    """
    if map_file and selected_dir:
        os.system(f"python main.py --map \"{map_file}\" --images_dir \"{selected_dir}\" --output_dir \"data/output\"")
        return Image.open("data/output/trajectory.jpg")
    return None


def clear_all():
    """
    Clear all loaded data and reset the interface.

    Returns:
        tuple: Empty values for the interface elements
    """
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


# Gradio interface realization
with gr.Blocks() as demo:
    gr.Markdown("# Zatulovskyi - Task 2: Classic CV - Drone navigation")

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