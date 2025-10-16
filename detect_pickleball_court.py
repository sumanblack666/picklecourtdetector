import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import xml.etree.ElementTree as ET

import cv2
import numpy as np

from pickleball_court_model import PickleballCourtModel
from utils import draw_court_model_to_img, draw_lines
from white_pixel_detector import WhitePixelDetector
from court_line_candidate_detector import CourtLineCandidateDetector
from model_fitting import ModelFitting


@dataclass(frozen=True)
class Keypoint:
    name: str
    x: float
    y: float
    visibility: int = 2


@dataclass
class DetectionResult:
    image_path: Path
    width: int
    height: int
    keypoints: List[Keypoint]
    best_model: np.ndarray
    score: float
    annotated_image: np.ndarray


def detect(image_path: Path, target_h: int = 960) -> DetectionResult:
    image_path = Path(image_path)
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Unable to load image at {image_path}")

    img = resize_img(img, target_h)

    white_pixel_detector = WhitePixelDetector()
    court_line_candidate_detector = CourtLineCandidateDetector()
    model_fitting = ModelFitting()

    best_model, score_max = court_model_init(
        img,
        white_pixel_detector,
        court_line_candidate_detector,
        model_fitting,
    )

    annotated_image = np.array(img)
    draw_lines(annotated_image, model_fitting.lines_horizontal)
    draw_lines(annotated_image, model_fitting.lines_vertical)
    draw_court_model_to_img(annotated_image, best_model)

    keypoints = project_model_keypoints(best_model)
    height, width = annotated_image.shape[:2]

    return DetectionResult(
        image_path=image_path,
        width=width,
        height=height,
        keypoints=keypoints,
        best_model=best_model,
        score=score_max,
        annotated_image=annotated_image,
    )


def process_image(
    image_path: Path,
    output_dir: Path,
    formats: Optional[Iterable[str]] = None,
    save_overlay: bool = True,
):
    result = detect(image_path)
    base_name = Path(image_path).stem
    return export_annotations(
        result,
        output_dir=output_dir,
        base_name=base_name,
        formats=formats,
        save_overlay=save_overlay,
    )


def export_annotations(
    result: DetectionResult,
    output_dir: Path,
    base_name: Optional[str] = None,
    formats: Optional[Iterable[str]] = None,
    save_overlay: bool = True,
) -> Dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = base_name or result.image_path.stem
    selected_formats = _normalise_formats(formats)

    outputs: Dict[str, Path] = {}

    if save_overlay:
        overlay_path = output_dir / f"{base_name}_overlay.png"
        cv2.imwrite(str(overlay_path), result.annotated_image)
        outputs["overlay"] = overlay_path

    for fmt in selected_formats:
        if fmt not in EXPORTERS:
            supported = ", ".join(EXPORTERS.keys())
            raise ValueError(f"Unsupported annotation format '{fmt}'. Supported formats: {supported}")
        outputs[fmt] = EXPORTERS[fmt](result, output_dir, base_name)

    return outputs


def generate_coco_annotations(result: DetectionResult) -> Dict[str, object]:
    image_id = _derive_image_id(result.image_path)
    keypoint_names = list(PickleballCourtModel.keypoints.keys())
    bbox = compute_bounding_box(result.keypoints)
    bbox_width = bbox[2] - bbox[0]
    bbox_height = bbox[3] - bbox[1]

    coco = {
        "info": {
            "description": "Pickleball Court Keypoint Detection",
            "version": "1.0",
            "year": datetime.now().year,
            "contributor": "Automated Pipeline",
            "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "licenses": [],
        "categories": [
            {
                "supercategory": "court",
                "id": 1,
                "name": "pickleball_court",
                "keypoints": keypoint_names,
                "skeleton": [],
            }
        ],
        "images": [
            {
                "id": image_id,
                "width": result.width,
                "height": result.height,
                "file_name": result.image_path.name,
                "license": 0,
                "flickr_url": "",
                "coco_url": "",
                "date_captured": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        ],
        "annotations": [],
    }

    keypoints_data: List[float] = []
    for kp in result.keypoints:
        keypoints_data.extend([float(kp.x), float(kp.y), int(kp.visibility)])

    coco["annotations"].append(
        {
            "id": image_id,
            "image_id": image_id,
            "category_id": 1,
            "bbox": [
                float(bbox[0]),
                float(bbox[1]),
                float(bbox_width),
                float(bbox_height),
            ],
            "area": float(bbox_width * bbox_height),
            "iscrowd": 0,
            "keypoints": keypoints_data,
            "num_keypoints": len(result.keypoints),
            "segmentation": [],
        }
    )

    return coco


def export_coco(result: DetectionResult, output_dir: Path, base_name: str) -> Path:
    coco_data = generate_coco_annotations(result)
    output_path = output_dir / f"{base_name}.coco.json"
    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(coco_data, fp, indent=4)
    return output_path


def export_yolo(result: DetectionResult, output_dir: Path, base_name: str) -> Path:
    xmin, ymin, xmax, ymax = compute_bounding_box(result.keypoints)

    if xmax - xmin <= 0:
        xmin, xmax = 0.0, float(result.width)
    if ymax - ymin <= 0:
        ymin, ymax = 0.0, float(result.height)

    box_width = xmax - xmin
    box_height = ymax - ymin
    x_center = xmin + box_width / 2.0
    y_center = ymin + box_height / 2.0

    def _norm_x(val: float) -> float:
        return min(max(val / float(result.width), 0.0), 1.0)

    def _norm_y(val: float) -> float:
        return min(max(val / float(result.height), 0.0), 1.0)

    yolo_values: List[float] = [
        0,
        _norm_x(x_center),
        _norm_y(y_center),
        float(box_width) / float(result.width),
        float(box_height) / float(result.height),
    ]

    for kp in result.keypoints:
        yolo_values.extend([
            _norm_x(kp.x),
            _norm_y(kp.y),
            int(kp.visibility),
        ])

    line_parts: List[str] = []
    for value in yolo_values:
        if isinstance(value, int):
            line_parts.append(str(value))
        else:
            line_parts.append(f"{value:.6f}")

    output_path = output_dir / f"{base_name}.txt"
    with open(output_path, "w", encoding="utf-8") as fp:
        fp.write(" ".join(line_parts) + "\n")
    return output_path


def export_cvat(result: DetectionResult, output_dir: Path, base_name: str) -> Path:
    root = ET.Element("annotations")
    meta = ET.SubElement(root, "meta")
    task = ET.SubElement(meta, "task")
    ET.SubElement(task, "id").text = "0"
    ET.SubElement(task, "name").text = base_name
    ET.SubElement(task, "size").text = "1"
    ET.SubElement(task, "mode").text = "annotation"
    ET.SubElement(task, "overlap").text = "0"
    labels = ET.SubElement(task, "labels")
    label = ET.SubElement(labels, "label")
    ET.SubElement(label, "name").text = "pickleball_court"

    points_attr = ";".join(f"{kp.x:.2f},{kp.y:.2f}" for kp in result.keypoints)

    image_el = ET.SubElement(
        root,
        "image",
        {
            "id": "0",
            "name": result.image_path.name,
            "width": str(result.width),
            "height": str(result.height),
        },
    )
    ET.SubElement(
        image_el,
        "points",
        {
            "label": "pickleball_court",
            "occluded": "0",
            "source": "auto",
            "points": points_attr,
            "z_order": "0",
        },
    )

    output_path = output_dir / f"{base_name}.cvat.xml"
    tree = ET.ElementTree(root)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return output_path


def export_keypoints_json(result: DetectionResult, output_dir: Path, base_name: str) -> Path:
    payload = {
        "image": {
            "path": str(result.image_path),
            "width": result.width,
            "height": result.height,
        },
        "keypoints": [
            {
                "name": kp.name,
                "x": float(kp.x),
                "y": float(kp.y),
                "visibility": int(kp.visibility),
            }
            for kp in result.keypoints
        ],
    }
    output_path = output_dir / f"{base_name}.keypoints.json"
    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=4)
    return output_path


EXPORTERS = {
    "coco": export_coco,
    "yolo": export_yolo,
    "cvat": export_cvat,
    "keypoints_json": export_keypoints_json,
}

SUPPORTED_FORMATS = tuple(EXPORTERS.keys())


def _normalise_formats(formats: Optional[Iterable[str]]) -> List[str]:
    if not formats:
        return ["coco"]

    seen = []
    for fmt in formats:
        fmt_lower = fmt.lower()
        if fmt_lower not in seen:
            seen.append(fmt_lower)
    return seen


def project_model_keypoints(best_model: np.ndarray) -> List[Keypoint]:
    keypoints: List[Keypoint] = []
    for name, (x, y) in PickleballCourtModel.keypoints.items():
        homogeneous = np.array([float(x), float(y), 1.0])
        projected = np.matmul(best_model, homogeneous)
        projected = projected / projected[2]
        keypoints.append(
            Keypoint(name=name, x=float(projected[0]), y=float(projected[1]))
        )
    return keypoints


def compute_bounding_box(keypoints: Sequence[Keypoint]) -> Tuple[float, float, float, float]:
    xs = [kp.x for kp in keypoints]
    ys = [kp.y for kp in keypoints]
    return min(xs), min(ys), max(xs), max(ys)


def _derive_image_id(image_path: Path) -> int:
    digits = "".join(ch for ch in image_path.stem if ch.isdigit())
    if digits:
        try:
            return int(digits)
        except ValueError:
            pass
    return 1


def main(
    image_path: Path,
    output_path: Path,
    formats: Optional[Iterable[str]] = None,
    save_overlay: bool = True,
):
    image_path = Path(image_path)
    output_path = Path(output_path)

    if output_path.suffix:
        output_dir = output_path.parent
        base_name = output_path.stem
    else:
        output_dir = output_path
        base_name = image_path.stem

    selected_formats = _normalise_formats(formats)
    result = detect(image_path)
    outputs = export_annotations(
        result,
        output_dir=output_dir,
        base_name=base_name,
        formats=selected_formats,
        save_overlay=save_overlay,
    )

    if output_path.suffix and selected_formats:
        primary_format = selected_formats[0]
        if primary_format in outputs:
            desired_path = output_path
            generated_path = outputs[primary_format]
            if generated_path != desired_path:
                desired_path.parent.mkdir(parents=True, exist_ok=True)
                generated_path.replace(desired_path)
                outputs[primary_format] = desired_path

    print("Best model:")
    print(result.best_model)
    print(f"Best score: {result.score}")
    for fmt, path in outputs.items():
        print(f"Saved {fmt} to {path}")

    return outputs


def court_model_init(img, whitePixelDetector, courtLineCandidateDetector, modelFitting):
    line_structure_const_and = whitePixelDetector.execute(img)
    lines_extended = courtLineCandidateDetector.execute(img, line_structure_const_and)
    best_model, score_max = modelFitting.execute(img, lines_extended, line_structure_const_and)
    return best_model, score_max


def resize_img(img, target_h=960):
    height, width, _ = img.shape
    if height > target_h:
        w_h_ratio = width / float(height)
        img = cv2.resize(img, (int(target_h * w_h_ratio), target_h), interpolation=cv2.INTER_AREA)
    return img


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect pickleball court keypoints and export annotations")
    parser.add_argument("image_path", type=Path, help="Path to the image to process")
    parser.add_argument(
        "output_path",
        type=Path,
        help="Output directory or file path (for legacy single-format exports)",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=SUPPORTED_FORMATS,
        default=["coco"],
        help="Annotation formats to export",
    )
    parser.add_argument(
        "--no-overlay",
        action="store_true",
        help="Do not save the visualization overlay image",
    )
    args = parser.parse_args()

    main(
        args.image_path,
        args.output_path,
        formats=args.formats,
        save_overlay=not args.no_overlay,
    )
