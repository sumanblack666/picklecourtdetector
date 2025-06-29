import numpy as np
import cv2
from line import Line
from config import HOUGH_THRESHOLD, THRESHOLD_R

class CourtLineCandidateDetector:
    def __init__(self, canny_low_thresh=50, canny_high_thresh=150, hough_rho=1, hough_theta=np.pi/180, hough_threshold=20, hough_min_line_length=20, hough_max_line_gap=5) -> None:
        self.canny_low_thresh = canny_low_thresh
        self.canny_high_thresh = canny_high_thresh
        self.hough_rho = hough_rho
        self.hough_theta = hough_theta
        self.hough_threshold = hough_threshold
        self.hough_min_line_length = hough_min_line_length
        self.hough_max_line_gap = hough_max_line_gap

    def execute(self, img, line_structure_const_and):
        self.img = np.array(img)
        self.line_structure_const_and = line_structure_const_and
        self.height, self.width = img.shape[:2]
        self.lines = {}

        # Use a more sensitive Canny edge detector
        self.blur_canny = cv2.Canny(self.line_structure_const_and, self.canny_low_thresh, self.canny_high_thresh)

        # Use Progressive Probabilistic Hough Transform for more robust line detection
        lines = cv2.HoughLinesP(
            self.blur_canny, 
            self.hough_rho, 
            self.hough_theta, 
            self.hough_threshold, 
            minLineLength=self.hough_min_line_length, 
            maxLineGap=self.hough_max_line_gap
        )

        if lines is not None:
            for i, line in enumerate(lines):
                x1, y1, x2, y2 = line[0]
                l = Line.from_two_point(i, (x1, y1), (x2, y2))
                self.lines[i] = l

        # Refine lines by merging close-by lines and extending them
        self.refined_lines = self._refine_lines(self.lines)

        return self.refined_lines

    def _refine_lines(self, lines):
        # This is a placeholder for a more advanced line refinement algorithm.
        # For now, we will just return the detected lines.
        # A more advanced implementation would merge collinear lines and 
        # remove duplicates.
        return lines