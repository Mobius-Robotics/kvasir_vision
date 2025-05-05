import cv2
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    # Your camera parameters
    camera_matrix = np.load("output/camera_matrix.npy")

    distortion_coeffs = np.load("output/dist_coeffs.npy")

    # Image dimensions
    image_width, image_height = 1920, 1080

    # Create a grid pattern for visualization
    def create_grid_image(width, height, grid_size=50):
        img = np.ones((height, width, 3), dtype=np.uint8) * 255

        # Draw vertical lines
        for x in range(0, width, grid_size):
            cv2.line(img, (x, 0), (x, height - 1), (0, 0, 0), 1)

        # Draw horizontal lines
        for y in range(0, height, grid_size):
            cv2.line(img, (0, y), (width - 1, y), (0, 0, 0), 1)

        return img

    # Create the grid image
    grid_img = create_grid_image(image_width, image_height)

    # Apply distortion to the grid image to visualize how the camera distorts
    mapx, mapy = cv2.initUndistortRectifyMap(
        camera_matrix,
        distortion_coeffs,
        None,
        None,
        (image_width, image_height),
        cv2.CV_32FC1,
    )

    # This shows what a perfect grid would look like when captured by your camera
    distorted_grid = cv2.remap(
        grid_img, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )

    # Undistort the grid to show what the camera correction does
    undistorted_grid = cv2.undistort(distorted_grid, camera_matrix, distortion_coeffs)

    # Create a heatmap visualization of distortion magnitude
    h, w = image_height, image_width
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    points = np.vstack((x.flatten(), y.flatten())).T
    points_undistorted = cv2.undistortPoints(
        points.reshape(-1, 1, 2), camera_matrix, distortion_coeffs, None, camera_matrix
    ).reshape(-1, 2)

    # Calculate displacement for each pixel
    displacement = np.sqrt(
        (points[:, 0] - points_undistorted[:, 0]) ** 2
        + (points[:, 1] - points_undistorted[:, 1]) ** 2
    )
    displacement_map = displacement.reshape(h, w)

    # Visualize results
    plt.figure(figsize=(20, 15))

    plt.subplot(2, 2, 1)
    plt.imshow(grid_img)
    plt.title("Original Grid")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.imshow(distorted_grid)
    plt.title("How Camera Would See Grid (With Distortion)")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.imshow(undistorted_grid)
    plt.title("Undistorted Grid")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.imshow(displacement_map, cmap="jet")
    plt.colorbar(label="Displacement (pixels)")
    plt.title("Distortion Magnitude Map")
    plt.axis("off")

    # Add circle indicating potential ROI where distortion is minimal
    center = (int(camera_matrix[0, 2]), int(camera_matrix[1, 2]))
    # Find a good ROI radius based on displacement map
    # This is a simple approach - you might want to adjust the threshold
    threshold = 1.0  # 1 pixel displacement threshold
    avg_displacement = np.mean(displacement_map, axis=1)
    roi_height = np.sum(avg_displacement < threshold)
    roi_radius = min(
        roi_height // 2,
        int(min(center[0], center[1], image_width - center[0], image_height - center[1])),
    )

    # Draw ROI circle on displacement map
    circle_img = displacement_map.copy()
    y, x = np.ogrid[:image_height, :image_width]
    mask = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= roi_radius**2
    plt.subplot(2, 2, 4)
    plt.imshow(displacement_map, cmap="jet")
    plt.contour(mask, levels=[0.5], colors="white", linewidths=2)
    plt.title(f"Distortion Map with Suggested ROI (r={roi_radius}px)")

    plt.tight_layout()
    plt.savefig("distortion_visualization.png", dpi=300)
    plt.show()

    # Print ROI suggestion
    print(f"Suggested ROI centered at {center} with radius {roi_radius} pixels")
    print(
        f"ROI rectangle: ({center[0] - roi_radius}, {center[1] - roi_radius}) to ({center[0] + roi_radius}, {center[1] + roi_radius})"
    )


if __name__ == "__main__":
    main()
