import numpy as np
from line import Line

class PickleballCourtModel:
    # The new keypoints are based on a reference image's pixel coordinates,
    # derived from the user's provided snippet.
    # We assume a top-down, undistorted view of the court.
    # The homography estimation will map this model to the actual image.

    # Reference dimensions from user snippet
    y_min = 561
    y_max = 2935
    court_height = y_max - y_min  # 2374

    # Calculated net position and NVZ offset based on standard court ratios
    net_y = int(y_min + (court_height / 2.0))      # 1748
    nvz_offset = int(court_height * (7.0 / 44.0))  # 377

    keypoints = {
        # Outer corners
        'top_left_corner': (286, 561),
        'top_right_corner': (1379, 561),
        'bottom_left_corner': (286, 2935),
        'bottom_right_corner': (1379, 2935),

        # NVZ corners
        'top_nvz_left': (286, net_y - nvz_offset),
        'top_nvz_right': (1379, net_y - nvz_offset),
        'bottom_nvz_left': (286, net_y + nvz_offset),
        'bottom_nvz_right': (1379, net_y + nvz_offset),

        # Centerline points
        'top_center_service_top': (832, 561),
        'bottom_center_service_bottom': (832, 2935),
        'top_center_service_bottom': (832, net_y - nvz_offset),
        'bottom_center_service_top': (832, net_y + nvz_offset),

        # Net posts
        'left_net_post': (286, net_y),
        'right_net_post': (1379, net_y)
    }

    # Create x and y arrays for the model fitting process
    x_coords = sorted(list(set([p[0] for p in keypoints.values()])))
    y_coords = sorted(list(set([p[1] for p in keypoints.values()])))
    x = np.array(x_coords)
    y = np.array(y_coords)

    # Define the court lines based on the new keypoints
    court_model_lines_h = [
        # Baselines
        Line.from_two_point(0, keypoints['top_left_corner'], keypoints['top_right_corner']),
        Line.from_two_point(1, keypoints['bottom_left_corner'], keypoints['bottom_right_corner']),
        # NVZ lines
        Line.from_two_point(2, keypoints['top_nvz_left'], keypoints['top_nvz_right']),
        Line.from_two_point(3, keypoints['bottom_nvz_left'], keypoints['bottom_nvz_right']),
        # Net line
        Line.from_two_point(4, keypoints['left_net_post'], keypoints['right_net_post'])
    ]

    court_model_lines_v = [
        # Sidelines
        Line.from_two_point(5, keypoints['top_left_corner'], keypoints['bottom_left_corner']),
        Line.from_two_point(6, keypoints['top_right_corner'], keypoints['bottom_right_corner']),
        # Centerline
        Line.from_two_point(7, keypoints['top_center_service_top'], keypoints['bottom_center_service_bottom'])
    ]