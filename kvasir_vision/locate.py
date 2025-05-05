import cv2
import numpy as np

from .common import ARUCO_DICT
from .common import MARKER_LEN_M as marker_size


def load_camera_intrinsics(camera_matrix_path: str, dist_coeffs_path: str):
    """
    Load camera intrinsics from .npy files.
    :param camera_matrix_path: Path to the NumPy file containing the camera matrix (3x3).
    :param dist_coeffs_path: Path to the NumPy file containing distortion coefficients.
    :return: camera matrix K and distortion coefficients dist.
    """
    K = np.load(camera_matrix_path)  # Load intrinsic matrix (fx, 0, cx; 0, fy, cy; 0, 0, 1)
    dist = np.load(dist_coeffs_path)  # Load distortion coefficients array
    return K, dist


def detect_and_refine_markers(image: np.ndarray, parameters):
    """
    Detect ArUco markers in an image and refine their corner positions.
    :param image: Input BGR image.
    :param parameters: ArUco detector parameters.
    :return: tuple(corners, ids) where corners is a list of 4x1x2 arrays, ids is Nx1 array.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Detect markers
    corners, ids, _ = cv2.aruco.detectMarkers(gray, ARUCO_DICT, parameters=parameters)
    # Subpixel refinement for noise reduction
    if corners:
        cv2.cornerSubPix(
            gray,
            np.concatenate(corners, axis=0),
            winSize=(3, 3),
            zeroZone=(-1, -1),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
    return corners, ids


def compute_camera_extrinsics(
    obj_points: np.ndarray, img_points: np.ndarray, K: np.ndarray, dist: np.ndarray
):
    """
    Solve PnP on known markers to find camera pose relative to the mat.
    :param obj_points: (N,3) array of known 3D points in mat frame.
    :param img_points: (N,2) array of corresponding 2D image points.
    :param K: Camera intrinsic matrix.
    :param dist: Distortion coefficients.
    :return: R_cm (3x3), t_cm (3x1) such that p_C = R_cm * p_M + t_cm.
    """
    _, rvec, tvec = cv2.solvePnP(obj_points, img_points, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    R_cm, _ = cv2.Rodrigues(rvec)  # Convert Rodrigues vector to rotation matrix
    t_cm = tvec.reshape(3, 1)
    return R_cm, t_cm


def approximate_unknown_corners(corners_2d, R_mc, t_mc, K, z0):
    """
    Back-project 2D corners into the mat frame by intersecting rays with plane z = z0.
    :param corners_2d: List of four (u,v) image points.
    :param R_mc: Rotation from camera to mat frame.
    :param t_mc: Translation from camera to mat frame.
    :param K: Intrinsic matrix.
    :param z0: Approximate height of unknown marker above mat.
    :return: (4,3) array of approximate 3D corner positions in mat frame.
    """
    R_cm_inv = R_mc.T
    t_cm_inv = -R_cm_inv @ t_mc
    approx_points = []

    for u, v in corners_2d:
        # Form ray in camera coords
        ray_c = np.linalg.inv(K) @ np.array([u, v, 1.0])
        # Transform ray to mat frame
        ray_m = R_cm_inv @ ray_c
        origin_m = t_cm_inv.flatten()
        # Solve for λ such that z = z0: origin_m[2] + λ * ray_m[2] = z0
        lam = (z0 - origin_m[2]) / ray_m[2]
        approx_point = origin_m + lam * ray_m
        approx_points.append(approx_point)

    return np.array(approx_points, dtype=np.float32)


def refine_unknown_pose(obj_points_approx, img_points, K, dist):
    """
    Refine the unknown marker pose with solvePnP on approximate 3D corners.
    :param obj_points_approx: (4,3) array of approximate 3D corners in mat frame.
    :param img_points: (4,2) array of detected image corners.
    :param K: Intrinsic matrix.
    :param dist: Distortion coefficients.
    :return: R_um (3x3), t_um (3x1) transform from unknown marker to camera frame.
    """
    _, rvec_u, tvec_u = cv2.solvePnP(
        obj_points_approx, img_points, K, dist, flags=cv2.SOLVEPNP_ITERATIVE
    )
    R_um, _ = cv2.Rodrigues(rvec_u)  # Marker→camera rotation
    t_um = tvec_u.reshape(3, 1)
    # Invert to get camera→marker if needed
    return R_um, t_um


def compose_and_extract_yaw(R_cm, t_cm, R_um, t_um):
    """
    Compose transforms to get unknown marker→mat pose and extract yaw.
    :param R_cm: Camera→mat rotation.
    :param t_cm: Camera→mat translation.
    :param R_um: Marker→camera rotation.
    :param t_um: Marker→camera translation.
    :return: (x, y, z, yaw_deg) pose in mat frame.
    """
    # Invert marker→camera to camera→marker
    R_mu = R_um.T
    t_mu = -R_mu @ t_um
    # Compose: marker→mat = camera→mat ∘ marker→camera
    R_mm = R_cm @ R_mu
    t_mm = R_cm @ t_mu + t_cm
    # Extract yaw around Z
    yaw = np.arctan2(R_mm[1, 0], R_mm[0, 0])
    yaw_deg = np.degrees(yaw)
    x, y, z = t_mm.flatten()
    return x, y, z, yaw_deg


def main():
    # Paths to your intrinsics
    K_path = "output/camera_matrix.npy"
    dist_path = "output/dist_coeffs.npy"

    # Load intrinsics
    K, dist = load_camera_intrinsics(K_path, dist_path)

    # Prepare ArUco detector
    parameters = cv2.aruco.DetectorParameters()

    # Read input image
    image = cv2.imread("input.jpg")
    corners, ids = detect_and_refine_markers(image, parameters)

    # -- Known markers (IDs and their 3D centers) --
    # Populate with your 7 known markers' center positions and size
    known_marker_info = {
        # id: (X, Y, Z), in mat frame
        0: (0.0, 0.0, 0.0),
        1: (0.1, 0.0, 0.0),
        # ... add all seven ...
    }
    obj_pts = []
    img_pts = []
    unknown_idx = None
    unknown_corners_2d = None

    # Separate known vs unknown
    for idx, c in zip(ids.flatten(), corners):
        pts_2d = c.reshape(-1, 2)
        if idx in known_marker_info:
            X, Y, Z = known_marker_info[idx]
            # Build 3D corners for known marker in mat frame
            for dx, dy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
                obj_pts.append([X + dx * marker_size / 2, Y + dy * marker_size / 2, Z])
            img_pts.extend(pts_2d)
        else:
            unknown_idx = idx
            unknown_corners_2d = pts_2d
    assert unknown_idx is not None and unknown_corners_2d is not None, "No unknown marker detected"

    obj_pts = np.array(obj_pts, dtype=np.float32)
    img_pts = np.array(img_pts, dtype=np.float32)

    # 1) Compute camera→mat extrinsics
    R_cm, t_cm = compute_camera_extrinsics(obj_pts, img_pts, K, dist)

    # 2) Approximate unknown corners at z0
    z0_approx = 43e-2  # meters
    approx_corners = approximate_unknown_corners(unknown_corners_2d, R_cm, t_cm, K, z0_approx)

    # 3) Refine unknown marker pose
    R_um, t_um = refine_unknown_pose(approx_corners, unknown_corners_2d, K, dist)

    # 4) Compose transforms & extract yaw
    x, y, z, yaw_deg = compose_and_extract_yaw(R_cm, t_cm, R_um, t_um)

    print(f"Unknown marker ID {unknown_idx}: X={x:.3f}, Y={y:.3f}, Z={z:.3f}, Yaw={yaw_deg:.1f}°")


if __name__ == "__main__":
    main()
