import math
import random
from pickleball_court_model import PickleballCourtModel

import cv2
import numpy as np

from line import Line
from config import CLS_ANGLE_THRESH

class ModelFitting:
    def __init__(self, ransac_iterations=1000, ransac_threshold=5.0) -> None:
        self.court_model_h, self.court_model_v, self.court_model_lines_h, self.court_model_lines_v = PickleballCourtModel.y, PickleballCourtModel.x, PickleballCourtModel.court_model_lines_h, PickleballCourtModel.court_model_lines_v
        self.ransac_iterations = ransac_iterations
        self.ransac_threshold = ransac_threshold

    def execute(self, img, lines_extended, line_structure_const_and):
        self.img = img
        self.lines_extended = lines_extended
        self.line_structure_const_and = line_structure_const_and
        self.height, self.width = self.line_structure_const_and.shape[:2]

        self.lines_horizontal, self.lines_vertical = self._find_line_correspondences()

        print(f'Num of lines_horizontal: {len(self.lines_horizontal)}')
        print(f'Num of lines_vertical: {len(self.lines_vertical)}')

        best_model, score_max = self._find_best_model_ransac()

        return best_model, score_max

    def _find_line_correspondences(self):
        lines_horizontal, lines_vertical = [], []
        degree_h_in_rad = CLS_ANGLE_THRESH * math.pi / 180

        for line in self.lines_extended.values():
            result_pyatan2 = math.atan2(line.m_y, line.m_x)
            if abs(result_pyatan2) < degree_h_in_rad:
                lines_horizontal.append(line)
            else:
                lines_vertical.append(line)

        def sort_dist_key(line_para, p1):
            return np.dot(line_para, p1)

        lines_horizontal.sort(
            key=lambda x: sort_dist_key(x.get_parameterized(), np.array([int(self.width / 2.0), 0, 1])), reverse=True
        )
        lines_vertical.sort(
            key=lambda x: sort_dist_key(x.get_parameterized(), np.array([0, int(self.height / 2.0), 1])), reverse=True
        )

        last = ModelFitting.unique(lines_horizontal)
        lines_horizontal = lines_horizontal[:last]
        
        last = ModelFitting.unique(lines_vertical)
        lines_vertical = lines_vertical[:last]

        return lines_horizontal, lines_vertical

    def _find_best_model_ransac(self):
        if len(self.lines_horizontal) < 2 or len(self.lines_vertical) < 2:
            return None, float('-inf')

        best_model = None
        score_max = float('-inf')

        def _line_distance_from_origin(line: Line) -> float:
            params = line.get_parameterized()
            return abs(params[2])

        model_corners = [
            PickleballCourtModel.keypoints['top_left_corner'],
            PickleballCourtModel.keypoints['top_right_corner'],
            PickleballCourtModel.keypoints['bottom_left_corner'],
            PickleballCourtModel.keypoints['bottom_right_corner'],
        ]
        pts_src = np.array([mc[:2] for mc in model_corners], dtype=np.float32)

        for _ in range(self.ransac_iterations):
            h_lines = sorted(random.sample(self.lines_horizontal, 2), key=_line_distance_from_origin)
            v_lines = sorted(random.sample(self.lines_vertical, 2), key=_line_distance_from_origin)

            p1 = Line.solve_intersection(h_lines[0], v_lines[0])
            p2 = Line.solve_intersection(h_lines[0], v_lines[1])
            p3 = Line.solve_intersection(h_lines[1], v_lines[0])
            p4 = Line.solve_intersection(h_lines[1], v_lines[1])

            points = [p1, p2, p3, p4]
            if any(p is None for p in points):
                continue

            pts_dest = np.stack(points).astype(np.float32)
            if not np.all(np.isfinite(pts_dest)):
                continue

            if not self._validate_quadrilateral(pts_dest):
                continue

            H, status = cv2.findHomography(pts_src, pts_dest, cv2.RANSAC, self.ransac_threshold)

            if H is None or status is None:
                continue

            score = self._evaluate_model(H)

            if score > score_max:
                score_max = score
                best_model = H

        return best_model, score_max

    def _validate_quadrilateral(self, pts):
        if pts.shape[0] != 4:
            return False
        
        for i in range(4):
            for j in range(i + 1, 4):
                dist = np.linalg.norm(pts[i] - pts[j])
                if dist < 10:
                    return False
        
        area = self._compute_quadrilateral_area(pts)
        if area < 1000:
            return False
        
        return True

    def _compute_quadrilateral_area(self, pts):
        x = pts[:, 0]
        y = pts[:, 1]
        area = 0.5 * abs(
            x[0] * y[1] - x[1] * y[0] +
            x[1] * y[2] - x[2] * y[1] +
            x[2] * y[3] - x[3] * y[2] +
            x[3] * y[0] - x[0] * y[3]
        )
        return area

    def _evaluate_model(self, H):
        score = 0
        trans_court_model = np.zeros((self.height, self.width), dtype=np.uint8)

        for line in self.court_model_lines_h + self.court_model_lines_v:
            start_pt_t = np.matmul(H, np.array([line.start_pt[0], line.start_pt[1], 1]))
            end_pt_t = np.matmul(H, np.array([line.end_pt[0], line.end_pt[1], 1]))
            start_pt_t /= start_pt_t[2]
            end_pt_t /= end_pt_t[2]

            cv2.line(trans_court_model, (int(start_pt_t[0]), int(start_pt_t[1])), (int(end_pt_t[0]), int(end_pt_t[1])), (255), 2)

        on_court = cv2.bitwise_and(trans_court_model, self.line_structure_const_and)
        score = np.count_nonzero(on_court)

        return score

    @staticmethod
    def unique(lines):
        if not lines:
            return 0
        if len(lines) == 1:
            return 1

        result = 0
        for i in range(1, len(lines)):
            if not (lines[result] == lines[i]):
                result += 1
                lines[result] = lines[i]
        return result + 1