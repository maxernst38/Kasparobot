import numpy as np
import cv2
import math

def warp_point(pt, H):
    """
    Warp a single (x, y) point into the rectified, scaled board image.
    Returns (x_px, y_px) in [0..800)×[0..800).
    """
    # build scale matrix S so that board-grid units [0..8] → pixels [0..800]
    S = np.array([
        [100,        0, 0],
        [       0, 100, 0],
        [       0,        0, 1],
    ], dtype=H.dtype)

    # compose and warp
    H_scaled = S @ H
    # use cv2.perspectiveTransform for one point
    pt_np = np.array([[[pt[0], pt[1]]]], dtype=np.float32)  # shape (1,1,2)
    warped = cv2.perspectiveTransform(pt_np, H_scaled.astype(np.float32))
    x_px, y_px = warped[0,0]
    return x_px, y_px

def map_detections_to_squares(detections, image_path, H):
    square_map = {}
    img = cv2.imread(image_path)
    img_h, img_w = img.shape[:2]

    for class_id, (cx, cy, w, h) in detections:
        # convert YOLO norm → pixel coords
        cx_px = cx * img_w
        cy_px = cy * img_h
        h_px  = h  * img_h

        # pick the bottom-center of the box
        x = cx_px
        y = cy_px + h_px/2

        warped_x, warped_y = warp_point((x, y), H)

        # now floor-divide by square_px to get [0..7]
        col = int(warped_x // 100)
        row = int(warped_y // 100) + 1

        if 0 <= col < 8 and 0 <= row < 8:
            file = chr(ord('a') + col)
            rank = 8 - row
            square = f"{file}{rank}"
            square_map[square] = class_id
        else:
            print(f"⚠️ Out of bounds: ({warped_x:.1f}, {warped_y:.1f}) → col={col}, row={row}")

    return square_map
