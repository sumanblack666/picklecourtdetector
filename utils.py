import cv2
import numpy as np

from pathlib import Path

from pickleball_court_model import PickleballCourtModel

def resize_img(img, target_h=960):
    height, width, _ = img.shape
    if height > target_h:
        w_h_ratio = width / float(height)
        img = cv2.resize(img, (int(target_h * w_h_ratio), target_h), interpolation=cv2.INTER_AREA)

    return img

def draw_lines(img, lines, color=(255, 0, 0)):
    for line in lines:
        line.draw_line_extended(img, color)


def draw_court_model_to_img(img, H):

    _draw_court_model_lines_to_img(img, H, PickleballCourtModel.court_model_lines_h)
    _draw_court_model_lines_to_img(img, H, PickleballCourtModel.court_model_lines_v)

    for name, point in PickleballCourtModel.keypoints.items():
        point_t = np.matmul(H, np.array([point[0], point[1], 1]))
        if abs(point_t[2]) < 1e-6:
            continue
        point_t = point_t / point_t[2]
        x, y = int(point_t[0]), int(point_t[1])
        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(img, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

    return img

def _draw_court_model_lines_to_img(img, H, court_model_lines):
    for line in court_model_lines:
        start_pt_t = np.matmul(H, np.array([line.start_pt[0], line.start_pt[1], 1]))
        end_pt_t = np.matmul(H, np.array([line.end_pt[0], line.end_pt[1], 1]))
        
        if abs(start_pt_t[2]) < 1e-6 or abs(end_pt_t[2]) < 1e-6:
            continue
            
        start_pt_t = start_pt_t / start_pt_t[2]
        end_pt_t = end_pt_t / end_pt_t[2]

        x1, y1 = int(start_pt_t[0]), int(start_pt_t[1])
        x2, y2 = int(end_pt_t[0]), int(end_pt_t[1])
        
        mid_pt = ((x1 + x2) // 2, (y1 + y2) // 2)
        
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 255), 3)
        
        if 0 <= mid_pt[0] < img.shape[1] and 0 <= mid_pt[1] < img.shape[0]:
            cv2.putText(img, str(line.id), mid_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)


def load_H_matrix(file_path:Path):
    with open(file_path, 'rb') as f:
        H_matrix = np.load(f)
        return H_matrix
    
def save_H_matrix(file_path, H_matrix):
    with open(file_path, 'wb') as f:
        np.save(f, H_matrix)