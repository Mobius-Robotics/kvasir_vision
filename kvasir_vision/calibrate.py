import glob
import os

import cv2
import numpy as np

from .common import ARUCO_DICT, BOARD

# ----------------------- 1. Load & Detect Corners ------------------------


def main() -> None:
    # Folder of captured calibration images
    INPUT_DIR = "output/calib_images"
    image_paths = glob.glob(os.path.join(INPUT_DIR, "*.png"))

    # Detector parameters
    detector_params = cv2.aruco.DetectorParameters()

    all_corners = []  # list of charuco corner arrays
    all_ids = []  # list of corresponding id arrays
    image_size = None

    for img_path in image_paths:
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if image_size is None:
            # width, height
            image_size = gray.shape[::-1]

        # Detect ArUco markers
        corners, ids, _ = cv2.aruco.detectMarkers(gray, ARUCO_DICT, parameters=detector_params)

        # Interpolate Charuco corners if markers detected
        if ids is not None and len(ids) > 0:
            _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                corners, ids, gray, BOARD
            )
            if charuco_ids is not None and len(charuco_ids) > 0:
                all_corners.append(charuco_corners)
                all_ids.append(charuco_ids)

    # ---------------------- 2. Camera Calibration ----------------------------

    print(f"[INFO] Detected {len(all_corners)} images with corners")
    print(f"[INFO] Detected {sum(len(ids) for ids in all_ids)} corners")

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        charucoCorners=all_corners,
        charucoIds=all_ids,
        board=BOARD,
        imageSize=image_size,
        cameraMatrix=None,
        distCoeffs=None,
    )
    print(f"[RESULT] Reprojection error: {ret}")
    print(f"Camera matrix:\n{camera_matrix}")
    print(f"Distortion coeffs:\n{dist_coeffs}")

    np.save("output/camera_matrix.npy", camera_matrix)
    np.save("output/dist_coeffs.npy", dist_coeffs)
    print("[SAVED] Camera matrix and distortion coefficients to output/")

    # ---------------- 3. Compute Optimal New Camera Matrix -------------------

    # We'll undistort each image with alpha=1 (keep all pixels)
    undistort_dir = "output/undistorted"
    os.makedirs(undistort_dir, exist_ok=True)

    for img_path in image_paths:
        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        # Refine camera matrix and get ROI (x, y, w, h)
        new_cam_mtx, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, (w, h), alpha=1.0, newImgSize=(w, h)
        )

        # Undistort and crop
        undistorted = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_cam_mtx)
        x, y, w_roi, h_roi = roi
        undistorted = undistorted[y : y + h_roi, x : x + w_roi]

        # Save result
        fname = os.path.join(undistort_dir, os.path.basename(img_path))
        cv2.imwrite(fname, undistorted)
        print(f"[SAVED] Undistorted image → {fname}")


if __name__ == "__main__":
    main()
