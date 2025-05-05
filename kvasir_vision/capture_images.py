import os
import time
from pathlib import Path

os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import cv2
from yapper import Yapper

from .common import ARUCO_DICT, BOARD
from .common import CAMERA_INDEX as CAMERA_ID
from .common import generate_board_image

CAPTURE_INTERVAL = 5.0  # seconds between captures
MIN_CHARUCO_CORNERS = 20  # minimum corners to accept frame
OUTPUT_DIR = Path("output/calib_images")


def speak(msg: str, *, yapper=Yapper(block=False)):
    print(msg)
    yapper.yap(msg)


def process_for_charuco(frame, gray, corners, ids, img_count) -> tuple[str, str | None]:
    "Process the frame for ChArUco corners and save the image if valid."

    # There should be at least one marker detected.
    if ids is None or len(ids) == 0:
        return "[INFO] No markers detected. Waiting...", None

    # Draw our detected markers and check for ChArUco corners.
    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
    _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, BOARD)
    corner_count = 0 if charuco_ids is None else len(charuco_ids)

    # Show some debugging text.
    cv2.putText(
        frame,
        f"Corners: {corner_count:2}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    # If we have enough corners, draw them and return the image path.
    if corner_count < MIN_CHARUCO_CORNERS:
        return (
            f"[INFO] Detected {corner_count} charuco corners; need ≥ {MIN_CHARUCO_CORNERS}. Waiting...",
            None,
        )
    cv2.aruco.drawDetectedCornersCharuco(frame, charuco_corners, charuco_ids)
    dest = str(OUTPUT_DIR / f"calib_{img_count:02d}.png")
    return f"[CAPTURED] Saved {dest} ({corner_count} corners)", dest


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    board_file = OUTPUT_DIR.parent / "charuco_board.png"
    cv2.imwrite(str(board_file), generate_board_image())
    speak(f"Saved Charuco board to '{board_file}'")

    detector_params = cv2.aruco.DetectorParameters()
    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    if not cap.isOpened():
        speak(f"[ERROR] Cannot open camera ID {CAMERA_ID}")
        return

    speak("[INFO] Starting video capture. Press 'q' to quit.")
    last_time = time.monotonic() - CAPTURE_INTERVAL
    img_count = 0

    while True:
        # Read a frame from the camera.
        ret, frame = cap.read()
        if not ret:
            speak("[ERROR] Failed to grab frame")
            break

        # Keep a copy of the original frame for saving.
        original_frame = frame.copy()

        # Search for ArUco markers in the grayscale frame.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, ARUCO_DICT, parameters=detector_params)

        # Process the frame for ChArUco corners.
        msg, dest = process_for_charuco(frame, gray, corners, ids, img_count)

        # Check if enough time has passed to capture a new image.
        now = time.monotonic()
        if now - last_time >= CAPTURE_INTERVAL:
            last_time = now
            if dest is not None:
                cv2.imwrite(dest, original_frame)
                img_count += 1
            speak(msg)

        # Display the frame with detected markers and corners, checking for 'q' to quit.
        frame = cv2.resize(frame, (960, 540))
        cv2.imshow("Charuco Calib Capture", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            speak("[INFO] Quitting.")
            break

    cap.release()
    cv2.destroyAllWindows()
    speak(f"[DONE] Captured {img_count} valid images into '{OUTPUT_DIR}/'.")


if __name__ == "__main__":
    main()
