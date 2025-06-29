import cv2
import numpy as np
from pathlib import Path
import json
import os
from datetime import datetime

# custom sub-classes
from pickleball_court_model import PickleballCourtModel
from utils import draw_court_model_to_img, draw_lines
from white_pixel_detector import WhitePixelDetector
from court_line_candidate_detector import CourtLineCandidateDetector
from model_fitting import ModelFitting

def main(image_path, output_path):
    img = cv2.imread(str(image_path))
    
    img = resize_img(img)

    # Init detectors
    whitePixelDetector = WhitePixelDetector()
    courtLineCandidateDetector = CourtLineCandidateDetector()
    modelFitting = ModelFitting()
    
    best_model, score_max = court_model_init(img, whitePixelDetector, courtLineCandidateDetector, modelFitting)


    # visualization
    img_1 = np.array(img)
    draw_lines(img_1, modelFitting.lines_horizontal)
    draw_lines(img_1, modelFitting.lines_vertical)

    # Draw the projected court model to the image
    result_img = np.array(img)
    draw_court_model_to_img(result_img, best_model)

    cv2.imwrite(output_path.replace(".json", ".png"), result_img)

    # Generate COCO annotations
    coco_annotations = generate_coco_annotations(image_path, img, best_model)
    with open(output_path, 'w') as f:
        json.dump(coco_annotations, f, indent=4)

    print('Best model:')
    print(best_model)
    print("Best score:", score_max)

def generate_coco_annotations(image_path, img, best_model):
    height, width, _ = img.shape
    image_id = int(os.path.splitext(os.path.basename(image_path))[0].replace("frame_", ""))

    annotations = {
        "info": {
            "description": "Pickleball Court Keypoint Detection",
            "version": "1.0",
            "year": 2023,
            "contributor": "Gemini",
            "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "licenses": [],
        "categories": [
            {
                "supercategory": "court",
                "id": 1,
                "name": "pickleball_court",
                "keypoints": list(PickleballCourtModel.keypoints.keys()),
                "skeleton": []
            }
        ],
        "images": [
            {
                "id": image_id,
                "width": width,
                "height": height,
                "file_name": os.path.basename(image_path),
                "license": 0,
                "flickr_url": "",
                "coco_url": "",
                "date_captured": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        ],
        "annotations": []
    }

    keypoints_data = []
    num_keypoints = 0
    for name, point in PickleballCourtModel.keypoints.items():
        point_t = np.matmul(best_model, np.array([point[0], point[1], 1]))
        point_t = point_t / point_t[2]
        keypoints_data.extend([float(point_t[0]), float(point_t[1]), 2])  # x, y, visibility (2: visible)
        num_keypoints += 1

    annotations["annotations"].append({
        "id": 1,
        "image_id": image_id,
        "category_id": 1,
        "bbox": [0, 0, width, height],  # Placeholder bbox, can be refined
        "area": width * height,  # Placeholder area
        "iscrowd": 0,
        "keypoints": keypoints_data,
        "num_keypoints": num_keypoints,
        "segmentation": [],
    })

    return annotations

    


def court_model_init(img, whitePixelDetector, courtLineCandidateDetector, modelFitting):
    ###################################
    # 3.1 White Pixel Detection
    ###################################
    
    line_structure_const_and = whitePixelDetector.execute(img)

    # court_line_candidate = whitePixelDetector.court_line_candidate
    # line_structure_const = whitePixelDetector.line_structure_const


    ###################################
    # 3.2 Court Line Candidate Detector
    ###################################

    lines_extended = courtLineCandidateDetector.execute(img, line_structure_const_and)

    # blur_canny = court_line_candidate_detector.blur_canny


    ###################################
    # 3.3 Model Fitting
    ###################################

    best_model, score_max = modelFitting.execute(img, lines_extended, line_structure_const_and)

    return best_model, score_max

def resize_img(img, target_h=960):
    height, width, _ = img.shape
    if height > target_h:
        w_h_ratio = width / float(height)
        img = cv2.resize(img, (int(target_h * w_h_ratio), target_h), interpolation=cv2.INTER_AREA)

    return img

if __name__ == '__main__':
    # Example usage when run as a standalone script
    test_image_path = Path('test_images/frame_00002.png')
    test_output_path = Path('output_annotations/frame_00002.json')
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(test_output_path), exist_ok=True)

    main(test_image_path, test_output_path)