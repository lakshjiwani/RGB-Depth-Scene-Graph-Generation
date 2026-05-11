"""
Converts 2D pixel detections + depth maps + camera poses
into 3D world-space coordinates.

Task Context (Task A):
    The task requires instantiating a 3D scene graph where nodes
    represent localized objects with 3D centroids or bounding boxes.

    This module provides the mathematical bridge between:
      - 2D bounding boxes from YOLOv8 (pixel space)
      - Depth PNG16 frames (per-pixel distance in metres)
      - Camera poses (4x4 world transform matrices)
      - Camera intrinsics (focal length, principal point)

    Pipeline per detection:
      1. Extract center pixel (u, v) from 2D bounding box
      2. Look up depth value d at (u, v) from depth frame
      3. Unproject (u, v, d) to 3D camera-space point using intrinsics
      4. Transform camera-space point to world-space using pose matrix

    Mathematical formulation:
      Camera space:
        X_cam = (u - cx) * d / fx
        Y_cam = (v - cy) * d / fy
        Z_cam = d

      World space:
        P_world = T_cam2world @ [X_cam, Y_cam, Z_cam, 1]^T

"""

import os
import json
import logging
import sys

import numpy as np
import cv2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# ── Data Classes ──────────────────────────────────────────────────────────────

class CameraIntrinsics:
    """
    Holds camera intrinsic parameters loaded from camera_info.json.

    Intrinsics define the lens geometry — how 3D points project onto
    the 2D image sensor. They are fixed for a given camera/render setup
    and are the same for both datasets (BasicHouse and Synagogue).

    Attributes:
        fx, fy : focal lengths in pixels
        cx, cy : principal point (optical centre) in pixels
        width  : image width in pixels
        height : image height in pixels
        near_m : minimum valid depth in metres
        far_m  : maximum valid depth in metres
    """

    def __init__(
        self,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        width: int,
        height: int,
        near_m: float = 0.05,
        far_m: float = 100.0,
    ):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.width = width
        self.height = height
        self.near_m = near_m
        self.far_m = far_m

    @classmethod
    def from_json(cls, json_path: str) -> "CameraIntrinsics":
        """
        Loads intrinsics from the camera_info.json file provided
        with each dataset.

        JSON format:
            {
                "width": 1024,
                "height": 768,
                "intrinsics": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
                "near_m": 0.05,
                "far_m": 100.0
            }

        Args:
            json_path: path to camera_info.json

        Returns:
            CameraIntrinsics instance
        """
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"camera_info.json not found: {json_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        K = data["intrinsics"]  # 3x3 matrix as nested list

        return cls(
            fx=float(K[0][0]),
            fy=float(K[1][1]),
            cx=float(K[0][2]),
            cy=float(K[1][2]),
            width=int(data["width"]),
            height=int(data["height"]),
            near_m=float(data.get("near_m", 0.05)),
            far_m=float(data.get("far_m", 100.0)),
        )

    @property
    def matrix(self) -> np.ndarray:
        """Returns the 3x3 intrinsic matrix K."""
        return np.array([
            [self.fx,      0.0, self.cx],
            [0.0,      self.fy, self.cy],
            [0.0,          0.0,     1.0],
        ])

    def __repr__(self) -> str:
        return (
            f"CameraIntrinsics("
            f"fx={self.fx:.2f}, fy={self.fy:.2f}, "
            f"cx={self.cx:.2f}, cy={self.cy:.2f}, "
            f"{self.width}x{self.height})"
        )


class PoseLoader:
    """
    Loads and parses camera poses from poses.txt.

    Each line in poses.txt contains 16 space-separated floats
    representing a 4x4 camera-to-world transformation matrix
    in row-major order.

    The matrix transforms a point from camera space to world space:
        P_world = T_cam2world @ P_camera

    task note:
        The task states poses are "already aligned to the global IFC
        coordinate system", meaning we can directly use these matrices
        to place detected objects in the same coordinate frame as the
        IFC building elements — this is what enables BIM fusion.
    """

    def __init__(self, pose_file: str):
        """
        Args:
            pose_file: path to pose/poses.txt
        """
        if not os.path.exists(pose_file):
            raise FileNotFoundError(f"poses.txt not found: {pose_file}")

        self.poses = self._load(pose_file)
        logger.info(f"Loaded {len(self.poses)} camera poses from {pose_file}")

    def _load(self, pose_file: str) -> list:
        """
        Parses poses.txt into a list of 4x4 numpy arrays.

        Each line = one frame's camera-to-world matrix (16 floats).
        """
        poses = []
        with open(pose_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                values = list(map(float, line.split()))
                if len(values) != 16:
                    logger.warning(
                        f"  Line {line_num}: expected 16 values, "
                        f"got {len(values)} — skipping"
                    )
                    continue
                # Reshape row-major 16 floats → 4x4 matrix
                matrix = np.array(values).reshape(4, 4)
                poses.append(matrix)
        return poses

    def __len__(self) -> int:
        return len(self.poses)

    def __getitem__(self, idx: int) -> np.ndarray:
        return self.poses[idx]

    def get(self, frame_idx: int) -> np.ndarray:
        """
        Returns the 4x4 camera-to-world matrix for a given frame index.

        Args:
            frame_idx: 0-based frame index

        Returns:
            4x4 numpy array (camera-to-world transform)
        """
        if frame_idx < 0 or frame_idx >= len(self.poses):
            raise IndexError(
                f"Frame index {frame_idx} out of range "
                f"(0 to {len(self.poses) - 1})"
            )
        return self.poses[frame_idx]


# ── Depth Frame Loading ───────────────────────────────────────────────────────

def load_depth_frame(depth_path: str, depth_scale: float = 1000.0) -> np.ndarray:
    """
    Loads a 16-bit PNG depth frame and converts to metres.

    The depth_png16 files store depth as 16-bit unsigned integers.
    Dividing by depth_scale (default 1000) converts to metres.
    This is the standard convention for depth cameras (e.g. RealSense).

    For the provided datasets:
        - Pixel value 0      → invalid / no depth
        - Pixel value 1000   → 1.0 metre
        - Pixel value 5000   → 5.0 metres

    Args:
        depth_path  : path to the .png depth file
        depth_scale : divide raw value by this to get metres

    Returns:
        depth_metres: float32 array of shape (H, W)
                      values are depth in metres, 0.0 = invalid
    """
    if not os.path.exists(depth_path):
        raise FileNotFoundError(f"Depth frame not found: {depth_path}")

    # IMREAD_UNCHANGED preserves the 16-bit values
    raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

    if raw is None:
        raise ValueError(f"Failed to read depth frame: {depth_path}")

    # If loaded as 3-channel (some 16-bit PNGs do this), take first channel
    # All channels are identical for depth images — no data loss
    if raw.ndim == 3:
        raw = raw[:, :, 0]

    # Convert to float32 metres
    depth_metres = raw.astype(np.float32) / depth_scale

    return depth_metres


def load_rgb_frame(rgb_path: str) -> np.ndarray:
    """
    Loads an RGB frame as a numpy array.

    Args:
        rgb_path: path to the .png RGB file

    Returns:
        RGB image as uint8 array (H, W, 3)
    """
    if not os.path.exists(rgb_path):
        raise FileNotFoundError(f"RGB frame not found: {rgb_path}")

    # OpenCV loads BGR by default — convert to RGB
    bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)

    if bgr is None:
        raise ValueError(f"Failed to read RGB frame: {rgb_path}")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


# ── Core 3D Projection Math ───────────────────────────────────────────────────

def pixel_to_camera_space(
    u: float,
    v: float,
    depth_m: float,
    intrinsics: CameraIntrinsics,
) -> np.ndarray:
    """
    Converts a 2D pixel coordinate + depth to a 3D point in camera space.

    This implements the standard pinhole camera back-projection:
        X_cam = (u - cx) * depth / fx
        Y_cam = (v - cy) * depth / fy
        Z_cam = depth

    Camera space convention (OpenCV / standard):
        +X → right
        +Y → down
        +Z → forward (into the scene)

    Args:
        u, v      : pixel coordinates (column, row)
        depth_m   : depth at (u, v) in metres
        intrinsics: camera intrinsic parameters

    Returns:
        3D point in camera space as shape (3,) array [X, Y, Z]
    """
    x_cam = (u - intrinsics.cx) * depth_m / intrinsics.fx
    y_cam = (v - intrinsics.cy) * depth_m / intrinsics.fy
    z_cam = depth_m
    return np.array([x_cam, y_cam, z_cam], dtype=np.float32)


def camera_to_world_space(
    point_cam: np.ndarray,
    pose_matrix: np.ndarray,
) -> np.ndarray:
    """
    Transforms a 3D point from camera space to world space.

    Uses the camera-to-world pose matrix (already aligned to IFC
    coordinate system as stated in the task).

    Math:
        P_world = T_cam2world @ [X_cam, Y_cam, Z_cam, 1]^T
        (homogeneous coordinates — the 1 enables translation)

    Args:
        point_cam   : 3D point in camera space, shape (3,)
        pose_matrix : 4x4 camera-to-world transform matrix

    Returns:
        3D point in world space, shape (3,)
    """
    # Homogeneous coordinates: append 1 to enable translation
    point_h = np.array([
        point_cam[0],
        point_cam[1],
        point_cam[2],
        1.0,
    ], dtype=np.float64)

    # Apply the 4x4 transform
    world_h = pose_matrix @ point_h

    # Return 3D point (drop the homogeneous w component)
    return world_h[:3]


def project_detection_to_3d(
    bbox_xyxy: tuple,
    depth_frame: np.ndarray,
    intrinsics: CameraIntrinsics,
    pose_matrix: np.ndarray,
    depth_scale: float = 1000.0,
    max_depth_m: float = 10.0,
    sample_radius: int = 5,
) -> np.ndarray | None:
    """
    Projects a 2D bounding box detection to a 3D world-space centroid.

    This is the core function called for every YOLOv8 detection.
    It implements the full pipeline:
        bbox pixels → center pixel → depth lookup → 3D camera → 3D world

    Depth sampling strategy:
        Rather than using a single center pixel (which may be noisy or
        on an object boundary), we sample a small patch of pixels
        around the center and take the median depth. Median is more
        robust than mean because it ignores outliers (holes, edges).

    Args:
        bbox_xyxy    : detection bounding box [x1, y1, x2, y2] in pixels
        depth_frame  : depth map in metres, shape (H, W)
        intrinsics   : camera intrinsic parameters
        pose_matrix  : 4x4 camera-to-world matrix for this frame
        depth_scale  : not used here (depth_frame already in metres)
        max_depth_m  : reject detections farther than this
        sample_radius: pixels around center to sample for median depth

    Returns:
        3D centroid in world space [X, Y, Z], or None if depth invalid
    """
    x1, y1, x2, y2 = bbox_xyxy

    # Center pixel of the bounding box
    u_center = (x1 + x2) / 2.0
    v_center = (y1 + y2) / 2.0

    # Clamp to image bounds
    h, w = depth_frame.shape
    u_int = int(np.clip(u_center, 0, w - 1))
    v_int = int(np.clip(v_center, 0, h - 1))

    # Sample a patch around the center for robust depth estimation
    u_lo = max(0, u_int - sample_radius)
    u_hi = min(w, u_int + sample_radius + 1)
    v_lo = max(0, v_int - sample_radius)
    v_hi = min(h, v_int + sample_radius + 1)

    depth_patch = depth_frame[v_lo:v_hi, u_lo:u_hi]

    # Filter out invalid depths (0 = no data, > max = too far)
    valid_depths = depth_patch[
        (depth_patch > 0.01) & (depth_patch < max_depth_m)
    ]

    if len(valid_depths) == 0:
        return None  # No valid depth at this detection

    # Median depth — robust against noise and boundary effects
    depth_m = float(np.median(valid_depths))

    # Step 1: pixel → camera space
    point_cam = pixel_to_camera_space(u_center, v_center, depth_m, intrinsics)

    # Step 2: camera space → world space
    point_world = camera_to_world_space(point_cam, pose_matrix)

    return point_world.astype(np.float32)


def get_depth_at_pixel(
    u: int,
    v: int,
    depth_frame: np.ndarray,
    radius: int = 3,
) -> float | None:
    """
    Returns median depth at a pixel location with a small sampling window.
    Used for single-point queries (e.g. keypoint localization).

    Args:
        u, v        : pixel coordinates
        depth_frame : depth map in metres
        radius      : sampling window half-size

    Returns:
        Median depth in metres, or None if invalid
    """
    h, w = depth_frame.shape
    u_lo = max(0, u - radius)
    u_hi = min(w, u + radius + 1)
    v_lo = max(0, v - radius)
    v_hi = min(h, v + radius + 1)

    patch = depth_frame[v_lo:v_hi, u_lo:u_hi]
    valid = patch[(patch > 0.01) & (patch < 100.0)]

    if len(valid) == 0:
        return None
    return float(np.median(valid))


# ── Frame Iterator ────────────────────────────────────────────────────────────

class FrameIterator:
    """
    Iterates over RGB + depth frame pairs for a dataset.

    Yields synchronized (rgb_frame, depth_frame, pose_matrix, frame_idx)
    tuples for processing by the object detector.

    Design decision:
        Frames are loaded one at a time (not pre-loaded into RAM) because
        383 frames × 1024×768 × 3 channels ≈ 900MB which exceeds typical
        RAM budgets. Lazy loading keeps memory usage constant.

    Args:
        rgb_dir    : directory containing rgb/000000.png etc.
        depth_dir  : directory containing depth_png16/000000.png etc.
        pose_loader: PoseLoader instance
        frame_step : process every Nth frame (1 = all frames)
        max_depth_m: reject depth values beyond this
    """

    def __init__(
        self,
        rgb_dir: str,
        depth_dir: str,
        pose_loader: PoseLoader,
        frame_step: int = 5,
        max_depth_m: float = 10.0,
    ):
        self.rgb_dir = rgb_dir
        self.depth_dir = depth_dir
        self.pose_loader = pose_loader
        self.frame_step = frame_step
        self.max_depth_m = max_depth_m

        # Discover all frame indices from RGB directory
        self.frame_indices = self._discover_frames()
        logger.info(
            f"FrameIterator: {len(self.frame_indices)} frames available "
            f"(step={frame_step}, "
            f"processing {len(list(range(0, len(self.frame_indices), frame_step)))})"
        )

    def _discover_frames(self) -> list:
        """Returns sorted list of frame indices found in rgb_dir."""
        indices = []
        if not os.path.exists(self.rgb_dir):
            logger.error(f"RGB directory not found: {self.rgb_dir}")
            return indices

        for fname in sorted(os.listdir(self.rgb_dir)):
            if fname.endswith(".png"):
                try:
                    idx = int(os.path.splitext(fname)[0])
                    indices.append(idx)
                except ValueError:
                    continue
        return indices

    def __len__(self) -> int:
        return len(range(0, len(self.frame_indices), self.frame_step))

    def __iter__(self):
        """
        Yields (rgb, depth, pose, frame_idx) for each sampled frame.
        Skips frames where depth or pose is unavailable.
        """
        for i in range(0, len(self.frame_indices), self.frame_step):
            frame_idx = self.frame_indices[i]

            # Build file paths
            rgb_path   = os.path.join(
                self.rgb_dir, f"{frame_idx:06d}.png"
            )
            depth_path = os.path.join(
                self.depth_dir, f"{frame_idx:06d}.png"
            )

            # Skip if pose unavailable for this frame
            if frame_idx >= len(self.pose_loader):
                logger.warning(f"  No pose for frame {frame_idx} — skipping")
                continue

            # Load frames
            try:
                rgb   = load_rgb_frame(rgb_path)
                depth = load_depth_frame(depth_path)
                pose  = self.pose_loader.get(frame_idx)
            except (FileNotFoundError, ValueError) as e:
                logger.warning(f"  Frame {frame_idx}: {e} — skipping")
                continue

            yield rgb, depth, pose, frame_idx


# ── Quick Test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("depth_projection.py — Self Test")
    print("=" * 60)

    camera_json = "data/BasicHouse_with_pc/camera_info.json"
    pose_file   = "data/BasicHouse_with_pc/pose/poses.txt"
    rgb_dir     = "data/BasicHouse_with_pc/rgb"
    depth_dir   = "data/BasicHouse_with_pc/depth_png16"

    if not os.path.exists(camera_json):
        print("Data not found. Please unzip BasicHouse_with_pc into data/")
        sys.exit(1)

    # ── Test 1: Camera intrinsics ─────────────────────────────────────────────
    print("\n[1] Loading camera intrinsics...")
    intrinsics = CameraIntrinsics.from_json(camera_json)
    print(f"    {intrinsics}")
    print(f"    K matrix:\n{intrinsics.matrix}")

    # ── Test 2: Pose loading ──────────────────────────────────────────────────
    print("\n[2] Loading camera poses...")
    poses = PoseLoader(pose_file)
    print(f"    Total poses: {len(poses)}")
    print(f"    Pose[0] (frame 0):\n{poses.get(0).round(4)}")
    print(f"    Pose[1] (frame 1):\n{poses.get(1).round(4)}")

    # ── Test 3: Depth frame loading ───────────────────────────────────────────
    print("\n[3] Loading first depth frame...")
    depth = load_depth_frame(os.path.join(depth_dir, "000000.png"))
    print(f"    Shape  : {depth.shape}")
    print(f"    dtype  : {depth.dtype}")
    print(f"    Min    : {depth.min():.4f} m")
    print(f"    Max    : {depth[depth > 0].max():.4f} m")
    print(f"    Median : {np.median(depth[depth > 0]):.4f} m")

    # ── Test 4: RGB frame loading ─────────────────────────────────────────────
    print("\n[4] Loading first RGB frame...")
    rgb = load_rgb_frame(os.path.join(rgb_dir, "000000.png"))
    print(f"    Shape : {rgb.shape}")
    print(f"    dtype : {rgb.dtype}")

    # ── Test 5: 3D projection math ────────────────────────────────────────────
    print("\n[5] Testing 3D projection math...")

    # Simulate a detection at image center
    h, w   = depth.shape
    u, v   = w // 2, h // 2
    pose_0 = poses.get(0)

    depth_val = get_depth_at_pixel(u, v, depth)
    print(f"    Center pixel  : ({u}, {v})")
    print(f"    Depth at center: {depth_val:.4f} m")

    if depth_val:
        cam_pt   = pixel_to_camera_space(u, v, depth_val, intrinsics)
        world_pt = camera_to_world_space(cam_pt, pose_0)
        print(f"    Camera space  : {cam_pt.round(4)}")
        print(f"    World space   : {world_pt.round(4)}")

    # ── Test 6: Full bbox projection ──────────────────────────────────────────
    print("\n[6] Testing full bbox projection...")

    # Simulate a bounding box in image center
    fake_bbox = (w // 2 - 50, h // 2 - 50, w // 2 + 50, h // 2 + 50)
    centroid = project_detection_to_3d(
        bbox_xyxy=fake_bbox,
        depth_frame=depth,
        intrinsics=intrinsics,
        pose_matrix=pose_0,
        max_depth_m=10.0,
    )
    if centroid is not None:
        print(f"    Fake bbox     : {fake_bbox}")
        print(f"    3D centroid   : {centroid.round(4)}")
    else:
        print("    No valid depth at bbox center")

    # ── Test 7: Frame iterator ────────────────────────────────────────────────
    print("\n[7] Testing FrameIterator (first 3 frames)...")
    iterator = FrameIterator(
        rgb_dir=rgb_dir,
        depth_dir=depth_dir,
        pose_loader=poses,
        frame_step=5,
    )
    print(f"    Total frames to process: {len(iterator)}")

    for count, (rgb, depth, pose, idx) in enumerate(iterator):
        print(
            f"    Frame {idx:04d} | "
            f"RGB {rgb.shape} | "
            f"Depth min={depth[depth > 0].min():.2f}m "
            f"max={depth[depth > 0].max():.2f}m | "
            f"Pose translation={pose[:3, 3].round(2)}"
        )
        if count >= 2:
            break

    print("\n" + "=" * 60)
    print("✅ depth_projection.py all tests passed!")
    print("=" * 60)