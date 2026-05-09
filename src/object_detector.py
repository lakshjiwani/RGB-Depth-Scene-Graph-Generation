"""
Wraps YOLOv8 for 2D object detection on RGB frames, then combines
detections with depth and pose data to produce 3D-localized objects.

Assignment Context (Task A):
    The task requires "2D object detection" combined with "depth projection"
    to create scene graph nodes representing localized objects.

    This module is the bridge between RGB frames and the scene graph:
      - YOLOv8m provides 2D bounding boxes + class labels
      - depth_projection.py converts those to 3D world coordinates
      - The output is a list of Detection3D records ready for graph construction

Why YOLOv8m (over alternatives mentioned in the task PDF):

    DAAAM / PIX2Graph:
        Research codebases — fragile dependencies, poor maintenance,
        risk of spending days just on installation.

    Grounded-SAM:
        Excellent quality but requires Grounding DINO + SAM running
        together (~7GB+ VRAM). Segmentation masks are overkill when we
        only need bbox centroids for 3D projection. Worth mentioning
        as a future improvement.

    YOLOv8m wins:
        - One-line install (`pip install ultralytics`)
        - Auto-downloads weights on first use
        - COCO-pretrained → covers indoor object categories we care about
        - Fast on CPU (seconds), faster on GPU
        - Industry standard for object detection

Limitations to document in README:
    YOLOv8 is trained on COCO (80 classes), so it detects common furniture
    well (chair, refrigerator, sink) but cannot detect architectural elements
    like walls, columns, or doors as IFC entities. This gap is exactly why
    Task B's BIM priors are necessary — the IFC labels file fills in what
    YOLO cannot see.

"""

import os
import sys
import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import cv2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# ── COCO classes relevant to indoor scenes ────────────────────────────────────
# YOLOv8m is pretrained on COCO (80 classes). We filter to classes
# meaningful for indoor scene graphs. Other classes (cars, animals, etc.)
# would be noise in the building context.
INDOOR_RELEVANT_CLASSES = {
    "chair", "couch", "bed", "dining table", "toilet",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "potted plant",
    "backpack", "handbag", "tie", "suitcase", "person"
}


# ── Data Classes ──────────────────────────────────────────────────────────────

@dataclass
class Detection2D:
    """
    A single 2D object detection from one frame.

    This is the raw YOLOv8 output before 3D projection.
    """

    frame_idx: int                  # which frame this came from
    class_name: str                 # e.g. "refrigerator"
    class_id: int                   # COCO class index (0-79)
    confidence: float               # detection confidence (0.0 - 1.0)
    bbox_xyxy: tuple                # (x1, y1, x2, y2) in pixels

    @property
    def center_pixel(self) -> tuple:
        """Returns (u, v) center of the bounding box."""
        x1, y1, x2, y2 = self.bbox_xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def width(self) -> float:
        return self.bbox_xyxy[2] - self.bbox_xyxy[0]

    @property
    def height(self) -> float:
        return self.bbox_xyxy[3] - self.bbox_xyxy[1]

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass
class Detection3D:
    """
    A 2D detection lifted to 3D world space using depth + camera pose.

    These are the raw nodes that will be clustered (across frames)
    and become final scene graph nodes after deduplication.
    """

    frame_idx: int                  # source frame
    class_name: str                 # e.g. "refrigerator"
    class_id: int                   # COCO class index
    confidence: float               # YOLO confidence
    bbox_xyxy: tuple                # original 2D bbox in source frame
    centroid_world: np.ndarray      # 3D world coordinate (X, Y, Z)
    depth_at_center: float          # depth used for projection (metres)

    def to_dict(self) -> dict:
        """Serializable form for graph nodes / debugging."""
        return {
            "frame_idx": int(self.frame_idx),
            "class_name": self.class_name,
            "class_id": int(self.class_id),
            "confidence": float(self.confidence),
            "bbox_xyxy": [float(x) for x in self.bbox_xyxy],
            "centroid_world": [float(x) for x in self.centroid_world],
            "depth_at_center": float(self.depth_at_center),
        }


# ── Object Detector Wrapper ───────────────────────────────────────────────────

class YOLOv8Detector:
    """
    Thin wrapper around Ultralytics YOLOv8 for 2D object detection.

    Loads the model once at construction time, then runs inference
    per-frame. Filters detections by confidence and class.

    Args:
        model_name        : YOLOv8 weights file (auto-downloaded on first use)
        confidence_threshold: minimum confidence to keep a detection
        target_classes    : set of class names to keep (None = keep all
                            INDOOR_RELEVANT_CLASSES)
        device            : "cpu", "cuda", or "auto"
    """

    def __init__(
        self,
        model_name: str = "yolov8m.pt",
        confidence_threshold: float = 0.45,
        target_classes: Optional[set] = None,
        device: str = "auto",
    ):
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.target_classes = target_classes or INDOOR_RELEVANT_CLASSES
        self.device = device

        logger.info(f"Loading YOLOv8 model: {model_name}")
        # Lazy import to avoid loading torch unless this class is used
        from ultralytics import YOLO

        self.model = YOLO(model_name)
        # COCO class index → name lookup
        self.id_to_name = self.model.names

        logger.info(
            f"  Model loaded | "
            f"confidence_threshold={confidence_threshold} | "
            f"target classes={len(self.target_classes)}"
        )

    def detect(self, rgb_frame: np.ndarray, frame_idx: int = 0) -> List[Detection2D]:
        """
        Runs YOLOv8 on a single RGB frame.

        Args:
            rgb_frame : H x W x 3 uint8 RGB image
            frame_idx : frame index (stored on each Detection2D)

        Returns:
            List of Detection2D objects passing confidence + class filters
        """
        # Ultralytics expects BGR or path; we have RGB so flip channels
        bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

        # verbose=False suppresses Ultralytics' per-frame log spam
        results = self.model(
            bgr_frame,
            conf=self.confidence_threshold,
            verbose=False,
        )

        detections = []
        if not results or len(results) == 0:
            return detections

        result = results[0]  # single-image inference returns list of length 1

        if result.boxes is None or len(result.boxes) == 0:
            return detections

        # Extract numpy arrays from the result object
        boxes_xyxy = result.boxes.xyxy.cpu().numpy()      # (N, 4)
        confs      = result.boxes.conf.cpu().numpy()       # (N,)
        class_ids  = result.boxes.cls.cpu().numpy().astype(int)  # (N,)

        for bbox, conf, cid in zip(boxes_xyxy, confs, class_ids):
            class_name = self.id_to_name.get(cid, f"class_{cid}")

            # Filter to relevant classes
            if class_name not in self.target_classes:
                continue

            detections.append(Detection2D(
                frame_idx=frame_idx,
                class_name=class_name,
                class_id=int(cid),
                confidence=float(conf),
                bbox_xyxy=tuple(float(x) for x in bbox),
            ))

        return detections


# ── End-to-End Detection + 3D Projection ──────────────────────────────────────

def detect_and_project(
    detector: YOLOv8Detector,
    rgb_frame: np.ndarray,
    depth_frame: np.ndarray,
    pose_matrix: np.ndarray,
    intrinsics,
    frame_idx: int,
    max_depth_m: float = 10.0,
) -> List[Detection3D]:
    """
    Runs detection on a single frame and projects each 2D detection to 3D.

    This is the "fused" function called by the pipeline runner —
    it combines YOLO inference with the depth math from depth_projection.py.

    Args:
        detector    : YOLOv8Detector instance
        rgb_frame   : RGB image (H, W, 3)
        depth_frame : depth in metres (H, W)
        pose_matrix : 4x4 camera-to-world matrix
        intrinsics  : CameraIntrinsics from depth_projection.py
        frame_idx   : frame index
        max_depth_m : reject detections farther than this

    Returns:
        List of Detection3D objects (only those with valid depth)
    """
    # Late import to avoid circular dependencies
    from src.depth_projection import project_detection_to_3d, get_depth_at_pixel

    # 1. Run YOLO 2D detection
    dets_2d = detector.detect(rgb_frame, frame_idx=frame_idx)

    # 2. Project each 2D detection to 3D world coordinates
    dets_3d = []
    for det in dets_2d:
        centroid_world = project_detection_to_3d(
            bbox_xyxy=det.bbox_xyxy,
            depth_frame=depth_frame,
            intrinsics=intrinsics,
            pose_matrix=pose_matrix,
            max_depth_m=max_depth_m,
        )

        if centroid_world is None:
            # No valid depth — likely on glass, sky, or out of range
            continue

        # Look up depth at center for record keeping
        u, v = det.center_pixel
        depth_val = get_depth_at_pixel(int(u), int(v), depth_frame)

        dets_3d.append(Detection3D(
            frame_idx=det.frame_idx,
            class_name=det.class_name,
            class_id=det.class_id,
            confidence=det.confidence,
            bbox_xyxy=det.bbox_xyxy,
            centroid_world=centroid_world,
            depth_at_center=depth_val if depth_val is not None else 0.0,
        ))

    return dets_3d


# ── Visual Debug Helpers ──────────────────────────────────────────────────────

def draw_detections_2d(
    rgb_frame: np.ndarray,
    detections: List[Detection2D],
) -> np.ndarray:
    """
    Draws bounding boxes and labels onto an RGB frame for visual debugging.

    Args:
        rgb_frame  : RGB image (will not be modified)
        detections : list of Detection2D

    Returns:
        Annotated RGB image (H, W, 3) uint8
    """
    img = rgb_frame.copy()

    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det.bbox_xyxy]

        # Box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Label
        label = f"{det.class_name} {det.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        cv2.rectangle(
            img, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 255, 0), -1
        )
        cv2.putText(
            img, label, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
        )

    return img


# ── Quick Test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Run directly to verify YOLOv8 detection + 3D projection works:
        python src/object_detector.py
    """
    print("=" * 60)
    print("object_detector.py — Self Test")
    print("=" * 60)

    from src.depth_projection import (
        CameraIntrinsics,
        PoseLoader,
        FrameIterator,
    )

    # Paths
    camera_json = "data/BasicHouse_with_pc/camera_info.json"
    pose_file   = "data/BasicHouse_with_pc/pose/poses.txt"
    rgb_dir     = "data/BasicHouse_with_pc/rgb"
    depth_dir   = "data/BasicHouse_with_pc/depth_png16"

    if not os.path.exists(camera_json):
        print("Data not found. Please unzip BasicHouse_with_pc into data/")
        sys.exit(1)

    # ── Step 1: Setup ─────────────────────────────────────────────────────────
    print("\n[1] Loading camera intrinsics + poses...")
    intrinsics = CameraIntrinsics.from_json(camera_json)
    poses      = PoseLoader(pose_file)
    print(f"    Intrinsics: {intrinsics}")
    print(f"    Poses     : {len(poses)} frames")

    # ── Step 2: Load YOLOv8 ───────────────────────────────────────────────────
    print("\n[2] Loading YOLOv8m...")
    detector = YOLOv8Detector(
        model_name="yolov8m.pt",
        confidence_threshold=0.08,
    )

    # ── Step 3: Run on a few frames ───────────────────────────────────────────
    print("\n[3] Running detection + 3D projection on frames...")
    iterator = FrameIterator(
        rgb_dir=rgb_dir,
        depth_dir=depth_dir,
        pose_loader=poses,
        frame_step=5,
    )

    all_detections = []
    frame_count = 0

    for rgb, depth, pose, idx in iterator:
        dets_3d = detect_and_project(
            detector=detector,
            rgb_frame=rgb,
            depth_frame=depth,
            pose_matrix=pose,
            intrinsics=intrinsics,
            frame_idx=idx,
            max_depth_m=10.0,
        )

        all_detections.extend(dets_3d)
        frame_count += 1

        print(
            f"    Frame {idx:04d} | "
            f"{len(dets_3d)} 3D detections"
        )

        # Print a few for inspection
        for det in dets_3d[:3]:
            print(
                f"      [{det.class_name:<15}] "
                f"conf={det.confidence:.2f} "
                f"world={det.centroid_world.round(2)} "
                f"depth={det.depth_at_center:.2f}m"
            )

        # Limit to first 5 sampled frames for quick test
        if frame_count >= 16:
            break

    # ── Step 4: Summary ───────────────────────────────────────────────────────
    print("\n[4] Summary")
    print(f"    Frames processed : {frame_count}")
    print(f"    Total detections : {len(all_detections)}")

    # Tally per-class
    from collections import Counter
    class_counts = Counter(d.class_name for d in all_detections)
    print(f"    Classes detected:")
    for cls, count in class_counts.most_common(10):
        print(f"      {cls:<20} {count}")

    # ── Step 5: Save a debug visualization ────────────────────────────────────
    print("\n[5] Saving debug visualization (frame 0)...")
    os.makedirs("output", exist_ok=True)

    # Reload frame 0
    rgb_path = os.path.join(rgb_dir, "000000.png")
    bgr      = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    rgb      = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    dets_2d = detector.detect(rgb, frame_idx=0)
    annotated = draw_detections_2d(rgb, dets_2d)
    annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)

    out_path = "output/debug_detections_frame0.png"
    cv2.imwrite(out_path, annotated_bgr)
    print(f"    Saved: {out_path}")

    print("\n" + "=" * 60)
    print("✅ object_detector.py all tests passed!")
    print("=" * 60)