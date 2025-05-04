import cv2
import numpy as np

# Size of paper in 
WIDTH_M, HEIGHT_M = 210e-3, 297e-3

# Printer margins
MARGIN_IN = 0.25
MARGIN_M = MARGIN_IN * 0.0254

# Printable area
PRINTABLE_W_M = WIDTH_M - 2 * MARGIN_M
PRINTABLE_H_M = HEIGHT_M - 2 * MARGIN_M

# Charuco board layout
SQUARES_Y = 14
SQUARE_LEN_M = PRINTABLE_H_M / SQUARES_Y
SQUARES_X = int(PRINTABLE_W_M / SQUARE_LEN_M)

MARKER_RATIO = 0.75
MARKER_LEN_M = SQUARE_LEN_M * MARKER_RATIO

# Aruco dictionary and Charuco board
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
BOARD = cv2.aruco.CharucoBoard(
    (SQUARES_X, SQUARES_Y),
    SQUARE_LEN_M,
    MARKER_LEN_M,
    ARUCO_DICT
)

DPI = 300
PX_W = int(WIDTH_M / 0.0254 * DPI)
PX_H = int(HEIGHT_M / 0.0254 * DPI)
MARGIN_PX = int(MARGIN_IN * DPI)
PRINTABLE_PX_W = PX_W - 2 * MARGIN_PX
PRINTABLE_PX_H = PX_H - 2 * MARGIN_PX

def generate_board_image():
    board_img = BOARD.generateImage(
        (PRINTABLE_PX_W, PRINTABLE_PX_H),
        marginSize=None,
        borderBits=1
    )

    # Create a white A4 canvas and embed the board
    page = np.full((PX_H, PX_W), 255, dtype=np.uint8)
    y0, x0 = MARGIN_PX, MARGIN_PX
    page[y0:y0 + PRINTABLE_PX_H, x0:x0 + PRINTABLE_PX_W] = board_img
    return page

CAMERA_INDEX = 2
