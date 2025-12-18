import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

def deduplicate_points(points, eps):
    #############################################
    # Merge clusters of points using DBSCAN at 'eps' distance
    #############################################

    if len(points) == 0:
        return []
    
    # Get clusters
    points_np = np.array(points, dtype=np.float32)
    clustering = DBSCAN(eps=eps, min_samples=1).fit(points_np)
    labels = clustering.labels_
    
    # Merge by centroid
    merged_points = []
    for label in set(labels):
        group = points_np[labels == label]
        centroid = np.mean(group, axis=0)
        merged_points.append(tuple(centroid))
    return merged_points

def compute_intersection(rho1, theta1, rho2, theta2):
    #############################################
    # Compute intersection of two lines given in (rho, theta) form
    #############################################

    # Convert to Cartesian
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    if np.linalg.matrix_rank(A) < 2:
        return None

    # Solve linear system
    x0, y0 = np.linalg.solve(A, b)
    return int(x0.item()), int(y0.item())

def sort_grid_geometrically(points, row_thresh=20):
    #############################################
    # Sort points into a grid based on geometric proximity
    #############################################

    points = sorted(points, key=lambda p: p[1])
    rows = []
    for pt in points:
        x, y = pt
        placed = False
        for row in rows:
            avg_y = np.mean([p[1] for p in row])
            if abs(y - avg_y) < row_thresh:
                row.append(pt)
                placed = True
                break
        if not placed:
            rows.append([pt])
    grid = [sorted(row, key=lambda p: p[0]) for row in rows]
    return grid

def compute_homography_from_grid(grid):
    #############################################
    # Compute homography from detected grid points to ideal chessboard coordinates
    #############################################

    # Get source and destination points
    src_pts = []
    dst_pts = []
    for i, row in enumerate(grid):
        for j, (x, y) in enumerate(row):
            src_pts.append([x, y])
            dst_pts.append([j, i]) 

    src_pts = np.array(src_pts, dtype=np.float32)
    dst_pts = np.array(dst_pts, dtype=np.float32)

    # Use cv2 to computer homography and its inverse
    H, _ = cv2.findHomography(src_pts, dst_pts)
    H_inv, _ = cv2.findHomography(dst_pts, src_pts)

    # Project grid points
    src_pts_h = np.concatenate([src_pts, np.ones((len(src_pts), 1))], axis=1)
    grid_projected = (H @ src_pts_h.T).T

    return grid_projected, H, H_inv

def create_mask(image):
    ##############################################
    # Create a binary mask to isolate the chessboard from the background
    ##############################################

    img = cv2.imread(image)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_green = np.array([50, 40, 40])
    upper_green = np.array([90, 255, 200])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel = np.ones((3, 3), np.uint8)
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)

    return mask_clean

def find_intersections(mask):
    #############################################
    # Detect lines using Hough Transform and compute their intersections
    #############################################
    
    edges = cv2.Canny(mask, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)
    
    if lines is None:
        raise ValueError("No lines detected")

    horizontal = []
    vertical = []
    for i, line in enumerate(lines):
        rho, theta = line[0]
        angle = np.degrees(theta) % 180
        if 80 < angle < 100:
            vertical.append((rho, theta))
        elif angle < 10 or angle > 170:
            horizontal.append((rho, theta))
    vertical = vertical[:14]
    horizontal = horizontal[:14]

    intersections = []
    for rho1, theta1 in horizontal:
        for rho2, theta2 in vertical:
            pt = compute_intersection(rho1, theta1, rho2, theta2)
            if pt is not None:
                intersections.append(pt)

    intersections = deduplicate_points(intersections, eps=50)

    return intersections