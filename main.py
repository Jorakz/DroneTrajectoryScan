import os
import cv2
import argparse
from tqdm import tqdm
from src.localization import DroneLocalizer
from src.trajectory import TrajectoryBuilder
from src.visualization import TrajectoryVisualizer



def parse_args():
    """
    Parse command line arguments for the drone trajectory reconstruction program.

    Returns:
        argparse.Namespace: Parsed command line arguments including:
            - map: Path to the global map image
            - images_dir: Directory containing drone images
            - output_dir: Directory for saving results
            - method: Feature extraction method (sift, orb, akaze)
            - smooth: Whether to apply trajectory smoothing
            - visualize_positions: Whether to visualize drone positions
            - view_boxes: Whether to display drone view bounding boxes
    """
    parser = argparse.ArgumentParser(description='Drone Trajectory Reconstruction from Images')
    parser.add_argument('--map', default='data/global_map.png',
                        help='Path to the global map image')
    parser.add_argument('--images_dir', default='data/crops',
                        help='Directory containing drone images')
    parser.add_argument('--output_dir', default='data/output',
                        help='Directory for saving results')
    parser.add_argument('--method', default='sift',
                        help='Feature extraction method (sift, orb, akaze)')
    parser.add_argument('--smooth', action='store_true',
                        help='Apply trajectory smoothing')
    parser.add_argument('--visualize_positions', action='store_true',
                        help='Visualize drone positions')
    parser.add_argument('--view_boxes', action='store_true',
                        help='Display drone view bounding boxes')
    return parser.parse_args()


def main():
    """
    Main function for drone trajectory reconstruction.

    This function:
    1. Loads the global map image
    2. Initializes the drone localizer
    3. Processes each drone image to find its position on the map
    4. Builds a trajectory from the detected positions
    5. Visualizes the results (trajectory, positions, view boxes)
    6. Saves the results to the specified output directory
    """
    args = parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load global map
    global_map = cv2.imread(args.map)
    if global_map is None:
        raise FileNotFoundError(f"Failed to load map: {args.map}")

    print(f"Global map size: {global_map.shape}")

    # Initialize drone localizer
    localizer = DroneLocalizer(global_map, feature_method=args.method)

    # Get list of drone images, sorted by number
    drone_images = sorted(
        [f for f in os.listdir(args.images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))],
        key=lambda name: int(name.split('_')[1].split('.')[0])
    )

    if not drone_images:
        raise FileNotFoundError(f"No images found in directory: {args.images_dir}")

    # Load one drone image to determine size
    sample_image_path = os.path.join(args.images_dir, drone_images[0])
    sample_drone_img = cv2.imread(sample_image_path)
    if sample_drone_img is None:
        raise RuntimeError("Failed to load sample drone image to determine size.")

    # Localize each drone image
    positions = []
    homographies = []

    print("Localizing drone images...")
    for img_name in tqdm(drone_images):
        img_path = os.path.join(args.images_dir, img_name)
        drone_img = cv2.imread(img_path)

        if drone_img is None:
            print(f"Skipping {img_name}: failed to load image")
            positions.append(None)
            homographies.append(None)
            continue

        position_and_homography = localizer.locate_drone_image(drone_img)
        if position_and_homography:
            position, homography = position_and_homography
            positions.append(position)
            homographies.append(homography)
        else:
            print(f"Failed to localize image: {img_name}")
            positions.append(None)
            homographies.append(None)

    # Build trajectory
    trajectory_builder = TrajectoryBuilder(smooth=args.smooth)
    trajectory = trajectory_builder.build_trajectory(positions)

    # Visualize results
    visualizer = TrajectoryVisualizer(global_map)

    # Draw trajectory
    trajectory_image = visualizer.draw_trajectory(trajectory)
    cv2.imwrite(os.path.join(args.output_dir, "trajectory.jpg"), trajectory_image)

    # Visualize positions
    if args.visualize_positions:
        positions_image = visualizer.draw_drone_positions(positions)
        cv2.imwrite(os.path.join(args.output_dir, "positions.jpg"), positions_image)

    # Display drone view boxes
    if args.view_boxes:
        boxes_image = visualizer.draw_drone_view_boxes(
            list(zip(positions, homographies)),
            drone_image_shape=(sample_drone_img.shape[0], sample_drone_img.shape[1])
        )
        cv2.imwrite(os.path.join(args.output_dir, "view_boxes.jpg"), boxes_image)

    print(f"Results saved to directory: {args.output_dir}")


if __name__ == "__main__":
    main()