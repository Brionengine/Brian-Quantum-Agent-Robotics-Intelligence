"""
Layer 4: Perception - Vision Pipeline

Implements the complete visual perception stack for Brian-QARI:
  - RGB image processing and object detection
  - Depth estimation and point cloud generation
  - Visual SLAM (Simultaneous Localization and Mapping)
  - Person detection with distance estimation
  - Hand/gesture recognition for human-robot interaction
  - Scene understanding and semantic segmentation

Designed to run on Brion Quantum's TPU infrastructure for real-time
performance, with CPU fallbacks for local development.

Integrates with BrianMind's perceive() pipeline:
    SensorReading (RGB/Depth) -> VisionPipeline.process() -> WorldState updates
"""

import math
import time
import logging
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# Data Structures
# ============================================================================

class ObjectClass(Enum):
    """Detectable object classes for robotic manipulation and navigation."""
    PERSON = 0
    HAND = 1
    FACE = 2
    CUP = 3
    BOTTLE = 4
    PLATE = 5
    BOWL = 6
    TOOL = 7
    BOX = 8
    CHAIR = 9
    TABLE = 10
    DOOR = 11
    HANDLE = 12
    BUTTON = 13
    SCREEN = 14
    PHONE = 15
    KEYBOARD = 16
    BALL = 17
    FRUIT = 18
    UNKNOWN = 99


@dataclass
class BoundingBox:
    """2D bounding box in image coordinates."""
    x_min: int
    y_min: int
    x_max: int
    y_max: int

    @property
    def center(self) -> Tuple[int, int]:
        return ((self.x_min + self.x_max) // 2, (self.y_min + self.y_max) // 2)

    @property
    def width(self) -> int:
        return self.x_max - self.x_min

    @property
    def height(self) -> int:
        return self.y_max - self.y_min

    @property
    def area(self) -> int:
        return self.width * self.height


@dataclass
class Detection:
    """A single object detection result."""
    object_class: ObjectClass
    confidence: float
    bbox: BoundingBox
    position_3d: Optional[np.ndarray] = None  # [x, y, z] in camera frame
    orientation_3d: Optional[np.ndarray] = None  # [qw, qx, qy, qz]
    velocity_3d: Optional[np.ndarray] = None  # estimated velocity
    track_id: int = -1  # for object tracking across frames
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PersonDetection(Detection):
    """Extended detection for people with pose and gesture info."""
    skeleton_2d: Optional[np.ndarray] = None   # (17, 2) keypoints
    skeleton_3d: Optional[np.ndarray] = None   # (17, 3) keypoints
    gesture: Optional[str] = None               # recognized gesture
    is_facing_robot: bool = False
    estimated_height: float = 0.0
    hand_state: str = "unknown"  # open, closed, pointing, waving


@dataclass
class PointCloud:
    """3D point cloud from depth sensor."""
    points: np.ndarray          # (N, 3) XYZ coordinates
    colors: Optional[np.ndarray] = None  # (N, 3) RGB colors
    normals: Optional[np.ndarray] = None  # (N, 3) surface normals
    timestamp: float = 0.0
    frame_id: str = "camera"


@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters."""
    fx: float           # focal length x (pixels)
    fy: float           # focal length y (pixels)
    cx: float           # principal point x
    cy: float           # principal point y
    width: int          # image width
    height: int         # image height
    distortion: np.ndarray = field(default_factory=lambda: np.zeros(5))  # k1,k2,p1,p2,k3


@dataclass
class SLAMPose:
    """Robot pose estimate from visual SLAM."""
    position: np.ndarray      # [x, y, z]
    orientation: np.ndarray   # [qw, qx, qy, qz]
    covariance: np.ndarray = field(default_factory=lambda: np.eye(6) * 0.01)
    timestamp: float = 0.0
    keyframe_id: int = -1
    num_tracked_features: int = 0
    tracking_quality: float = 0.0  # 0-1


@dataclass
class OccupancyCell:
    """Single cell in the occupancy grid."""
    probability: float = 0.5  # 0=free, 0.5=unknown, 1=occupied
    height: float = 0.0
    semantic_label: int = 0
    last_updated: float = 0.0


# ============================================================================
# Object Detector
# ============================================================================

class ObjectDetector:
    """
    Real-time object detection optimized for robotic manipulation.

    Uses a lightweight detection backbone with robot-relevant object classes.
    Supports YOLO-family models via ultralytics, with fallback to
    template matching for environments without GPU.
    """

    def __init__(self, model_path: Optional[str] = None,
                 confidence_threshold: float = 0.5,
                 nms_threshold: float = 0.45,
                 device: str = "cpu"):
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.device = device
        self._model = None
        self._model_path = model_path
        self._class_map: Dict[int, ObjectClass] = {}
        self._initialized = False

        # Try to load detection model
        self._try_load_model(model_path)

    def _try_load_model(self, model_path: Optional[str]) -> None:
        """Attempt to load YOLO model, fall back to built-in detector."""
        try:
            from ultralytics import YOLO
            path = model_path or "yolov8n.pt"
            self._model = YOLO(path)
            self._build_class_map()
            self._initialized = True
            logger.info(f"ObjectDetector loaded YOLO model: {path}")
        except ImportError:
            logger.info("ultralytics not available, using geometric detector fallback")
            self._initialized = True
        except Exception as e:
            logger.warning(f"Failed to load YOLO model: {e}, using fallback")
            self._initialized = True

    def _build_class_map(self) -> None:
        """Map COCO class indices to our ObjectClass enum."""
        coco_to_brian = {
            0: ObjectClass.PERSON,
            39: ObjectClass.BOTTLE,
            41: ObjectClass.CUP,
            42: ObjectClass.PLATE,  # fork -> plate (close enough)
            45: ObjectClass.BOWL,
            56: ObjectClass.CHAIR,
            60: ObjectClass.TABLE,
            67: ObjectClass.PHONE,
            73: ObjectClass.KEYBOARD,
            32: ObjectClass.BALL,  # sports ball
            46: ObjectClass.FRUIT,  # banana
            47: ObjectClass.FRUIT,  # apple
        }
        self._class_map = coco_to_brian

    def detect(self, rgb_image: np.ndarray,
               depth_image: Optional[np.ndarray] = None,
               intrinsics: Optional[CameraIntrinsics] = None) -> List[Detection]:
        """
        Detect objects in an RGB image.

        Args:
            rgb_image: (H, W, 3) uint8 RGB image.
            depth_image: (H, W) float32 depth in meters (optional).
            intrinsics: Camera intrinsics for 3D position estimation.

        Returns:
            List of Detection objects.
        """
        if not self._initialized:
            return []

        detections = []

        if self._model is not None:
            detections = self._detect_yolo(rgb_image)
        else:
            detections = self._detect_geometric(rgb_image)

        # Add 3D positions from depth
        if depth_image is not None and intrinsics is not None:
            for det in detections:
                det.position_3d = self._estimate_3d_position(
                    det.bbox, depth_image, intrinsics)

        return detections

    def _detect_yolo(self, rgb_image: np.ndarray) -> List[Detection]:
        """Run YOLO inference."""
        results = self._model(rgb_image, verbose=False,
                              conf=self.confidence_threshold,
                              iou=self.nms_threshold)
        detections = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                obj_class = self._class_map.get(cls_id, ObjectClass.UNKNOWN)
                detections.append(Detection(
                    object_class=obj_class,
                    confidence=conf,
                    bbox=BoundingBox(x1, y1, x2, y2),
                ))
        return detections

    def _detect_geometric(self, rgb_image: np.ndarray) -> List[Detection]:
        """
        Fallback detector using color segmentation and contour analysis.
        Useful for simulation environments or when YOLO is unavailable.
        """
        detections = []
        h, w = rgb_image.shape[:2]

        # Convert to HSV for color-based detection
        # Simple approach: find regions of distinct color
        gray = np.mean(rgb_image, axis=2).astype(np.uint8)

        # Edge detection via Sobel approximation
        dx = np.abs(np.diff(gray.astype(float), axis=1))
        dy = np.abs(np.diff(gray.astype(float), axis=0))

        # Pad to original size
        dx = np.pad(dx, ((0, 0), (0, 1)), mode='edge')
        dy = np.pad(dy, ((0, 1), (0, 0)), mode='edge')

        edges = np.sqrt(dx**2 + dy**2)
        edge_thresh = np.percentile(edges, 90)
        binary = (edges > edge_thresh).astype(np.uint8)

        # Find connected components (simple flood fill approach)
        # For efficiency, use block-based analysis
        block_size = 32
        for by in range(0, h - block_size, block_size):
            for bx in range(0, w - block_size, block_size):
                block = binary[by:by+block_size, bx:bx+block_size]
                edge_density = np.mean(block)
                if edge_density > 0.15:
                    # Potential object region - expand to find bounds
                    x1, y1, x2, y2 = self._expand_region(
                        binary, bx, by, block_size, w, h)
                    area = (x2 - x1) * (y2 - y1)
                    if area > 500 and area < (w * h * 0.5):
                        detections.append(Detection(
                            object_class=ObjectClass.UNKNOWN,
                            confidence=min(edge_density * 2.0, 0.8),
                            bbox=BoundingBox(x1, y1, x2, y2),
                        ))

        # Non-maximum suppression
        detections = self._nms(detections)
        return detections[:20]  # limit to top 20

    def _expand_region(self, binary: np.ndarray,
                       bx: int, by: int, block_size: int,
                       w: int, h: int) -> Tuple[int, int, int, int]:
        """Expand a detected region to its full extent."""
        x1, y1 = bx, by
        x2, y2 = min(bx + block_size, w), min(by + block_size, h)

        # Expand right
        while x2 < w - 1 and np.mean(binary[y1:y2, x2:min(x2+8, w)]) > 0.05:
            x2 = min(x2 + 8, w)
        # Expand down
        while y2 < h - 1 and np.mean(binary[y2:min(y2+8, h), x1:x2]) > 0.05:
            y2 = min(y2 + 8, h)
        # Expand left
        while x1 > 0 and np.mean(binary[y1:y2, max(x1-8, 0):x1]) > 0.05:
            x1 = max(x1 - 8, 0)
        # Expand up
        while y1 > 0 and np.mean(binary[max(y1-8, 0):y1, x1:x2]) > 0.05:
            y1 = max(y1 - 8, 0)

        return x1, y1, x2, y2

    def _nms(self, detections: List[Detection]) -> List[Detection]:
        """Simple non-maximum suppression."""
        if not detections:
            return []
        detections.sort(key=lambda d: d.confidence, reverse=True)
        keep = []
        for det in detections:
            overlap = False
            for kept in keep:
                iou = self._compute_iou(det.bbox, kept.bbox)
                if iou > self.nms_threshold:
                    overlap = True
                    break
            if not overlap:
                keep.append(det)
        return keep

    def _compute_iou(self, a: BoundingBox, b: BoundingBox) -> float:
        x1 = max(a.x_min, b.x_min)
        y1 = max(a.y_min, b.y_min)
        x2 = min(a.x_max, b.x_max)
        y2 = min(a.y_max, b.y_max)
        if x2 <= x1 or y2 <= y1:
            return 0.0
        intersection = (x2 - x1) * (y2 - y1)
        union = a.area + b.area - intersection
        return intersection / union if union > 0 else 0.0

    def _estimate_3d_position(self, bbox: BoundingBox,
                               depth_image: np.ndarray,
                               intrinsics: CameraIntrinsics) -> np.ndarray:
        """Estimate 3D position of detected object from depth map."""
        cx_obj, cy_obj = bbox.center

        # Sample depth in the center region of the bounding box
        margin_x = bbox.width // 4
        margin_y = bbox.height // 4
        roi = depth_image[
            max(bbox.y_min + margin_y, 0):min(bbox.y_max - margin_y, depth_image.shape[0]),
            max(bbox.x_min + margin_x, 0):min(bbox.x_max - margin_x, depth_image.shape[1]),
        ]

        # Use median depth (robust to noise and partial occlusion)
        valid_depths = roi[(roi > 0.1) & (roi < 10.0)]
        if len(valid_depths) == 0:
            return np.array([0.0, 0.0, 0.0])
        z = float(np.median(valid_depths))

        # Back-project to 3D using pinhole model
        x = (cx_obj - intrinsics.cx) * z / intrinsics.fx
        y = (cy_obj - intrinsics.cy) * z / intrinsics.fy

        return np.array([x, y, z])


# ============================================================================
# Depth Processor
# ============================================================================

class DepthProcessor:
    """
    Processes depth images into point clouds and spatial representations.
    Supports depth cameras (RealSense, ZED) and computed stereo depth.
    """

    def __init__(self, intrinsics: CameraIntrinsics,
                 min_depth: float = 0.1,
                 max_depth: float = 10.0,
                 voxel_size: float = 0.02):
        self.intrinsics = intrinsics
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.voxel_size = voxel_size

        # Precompute pixel-to-ray lookup table
        self._u_map, self._v_map = np.meshgrid(
            np.arange(intrinsics.width), np.arange(intrinsics.height))
        self._x_factor = (self._u_map - intrinsics.cx) / intrinsics.fx
        self._y_factor = (self._v_map - intrinsics.cy) / intrinsics.fy

    def depth_to_pointcloud(self, depth_image: np.ndarray,
                            rgb_image: Optional[np.ndarray] = None,
                            transform: Optional[np.ndarray] = None) -> PointCloud:
        """
        Convert a depth image to a 3D point cloud.

        Args:
            depth_image: (H, W) float32 depth in meters.
            rgb_image: (H, W, 3) uint8 for colored point cloud.
            transform: 4x4 transformation to apply (camera to world).

        Returns:
            PointCloud with 3D points and optional colors.
        """
        # Filter valid depths
        valid = (depth_image > self.min_depth) & (depth_image < self.max_depth)

        z = depth_image[valid]
        x = self._x_factor[valid] * z
        y = self._y_factor[valid] * z

        points = np.stack([x, y, z], axis=-1)

        # Apply transform if provided
        if transform is not None:
            R = transform[:3, :3]
            t = transform[:3, 3]
            points = (R @ points.T).T + t

        colors = None
        if rgb_image is not None:
            colors = rgb_image[valid].astype(np.float32) / 255.0

        return PointCloud(
            points=points.astype(np.float32),
            colors=colors,
            timestamp=time.time(),
        )

    def estimate_normals(self, cloud: PointCloud, k_neighbors: int = 20) -> PointCloud:
        """
        Estimate surface normals for each point using local PCA.
        Uses a fast grid-based neighbor search.
        """
        points = cloud.points
        n = len(points)
        if n < k_neighbors:
            cloud.normals = np.tile([0, 0, 1], (n, 1)).astype(np.float32)
            return cloud

        normals = np.zeros_like(points)

        # Voxel grid for fast neighbor lookup
        grid = {}
        vs = self.voxel_size * 5  # search radius
        for i, pt in enumerate(points):
            key = (int(pt[0] / vs), int(pt[1] / vs), int(pt[2] / vs))
            if key not in grid:
                grid[key] = []
            grid[key].append(i)

        for i, pt in enumerate(points):
            key = (int(pt[0] / vs), int(pt[1] / vs), int(pt[2] / vs))
            # Gather neighbors from adjacent voxels
            neighbors = []
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    for dz in range(-1, 2):
                        nkey = (key[0]+dx, key[1]+dy, key[2]+dz)
                        if nkey in grid:
                            neighbors.extend(grid[nkey])

            if len(neighbors) < 3:
                normals[i] = [0, 0, 1]
                continue

            # Select k nearest from candidates
            neighbor_pts = points[neighbors[:min(len(neighbors), k_neighbors * 3)]]
            dists = np.linalg.norm(neighbor_pts - pt, axis=1)
            nearest_idx = np.argsort(dists)[:k_neighbors]
            local_pts = neighbor_pts[nearest_idx]

            # PCA: smallest eigenvector = normal
            centroid = np.mean(local_pts, axis=0)
            centered = local_pts - centroid
            cov = centered.T @ centered / len(centered)
            try:
                eigenvalues, eigenvectors = np.linalg.eigh(cov)
                normal = eigenvectors[:, 0]  # smallest eigenvalue
                # Orient toward camera (z-axis pointing convention)
                if normal[2] > 0:
                    normal = -normal
                normals[i] = normal
            except np.linalg.LinAlgError:
                normals[i] = [0, 0, 1]

        cloud.normals = normals.astype(np.float32)
        return cloud

    def compute_floor_plane(self, cloud: PointCloud,
                            height_threshold: float = 0.05) -> Tuple[np.ndarray, float]:
        """
        Detect the floor plane using RANSAC.

        Returns:
            (normal_vector, distance_from_origin) of the floor plane.
        """
        points = cloud.points
        if len(points) < 100:
            return np.array([0.0, 1.0, 0.0]), 0.0

        best_inliers = 0
        best_normal = np.array([0.0, 1.0, 0.0])
        best_d = 0.0

        # RANSAC iterations
        n_iterations = 100
        rng = np.random.RandomState(42)

        for _ in range(n_iterations):
            # Sample 3 random points
            idx = rng.choice(len(points), 3, replace=False)
            p1, p2, p3 = points[idx[0]], points[idx[1]], points[idx[2]]

            # Compute plane normal
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            norm = np.linalg.norm(normal)
            if norm < 1e-10:
                continue
            normal /= norm

            # Plane equation: n . x + d = 0
            d = -np.dot(normal, p1)

            # Count inliers
            distances = np.abs(points @ normal + d)
            n_inliers = np.sum(distances < height_threshold)

            if n_inliers > best_inliers:
                best_inliers = n_inliers
                best_normal = normal
                best_d = d

        # Ensure normal points upward
        if best_normal[1] < 0:
            best_normal = -best_normal
            best_d = -best_d

        return best_normal, best_d

    def compute_obstacle_map(self, cloud: PointCloud,
                             floor_height: float = 0.0,
                             resolution: float = 0.05,
                             map_size: float = 5.0) -> np.ndarray:
        """
        Generate a 2D obstacle occupancy grid from a point cloud.

        Args:
            cloud: Input point cloud.
            floor_height: Height of the floor plane.
            resolution: Grid cell size in meters.
            map_size: Side length of the square grid in meters.

        Returns:
            2D occupancy grid (0=free, 1=occupied, 0.5=unknown).
        """
        grid_size = int(map_size / resolution)
        grid = np.full((grid_size, grid_size), 0.5, dtype=np.float32)

        points = cloud.points
        # Filter to points above floor and below max height
        above_floor = points[:, 1] > (floor_height + 0.05)
        below_ceiling = points[:, 1] < (floor_height + 2.0)
        valid = above_floor & below_ceiling
        obstacle_pts = points[valid]

        if len(obstacle_pts) == 0:
            return grid

        # Project to 2D grid (x-z plane, y is up)
        half_size = map_size / 2.0
        for pt in obstacle_pts:
            gx = int((pt[0] + half_size) / resolution)
            gz = int((pt[2] + half_size) / resolution)
            if 0 <= gx < grid_size and 0 <= gz < grid_size:
                grid[gz, gx] = min(grid[gz, gx] + 0.3, 1.0)

        # Mark free space along camera rays
        origin = np.array([0, 0, 0])
        for pt in points[~valid]:
            gx = int((pt[0] + half_size) / resolution)
            gz = int((pt[2] + half_size) / resolution)
            if 0 <= gx < grid_size and 0 <= gz < grid_size:
                grid[gz, gx] = max(grid[gz, gx] - 0.1, 0.0)

        return grid


# ============================================================================
# Visual SLAM (Simplified Feature-Based)
# ============================================================================

class VisualSLAM:
    """
    Lightweight visual SLAM for robot localization and mapping.

    Uses feature extraction and matching to track camera motion and build
    a sparse 3D map. Designed to work alongside depth cameras for
    metric accuracy.

    This is a simplified implementation suitable for indoor environments.
    For production, replace with ORB-SLAM3 or OpenVSLAM integration.
    """

    def __init__(self, intrinsics: CameraIntrinsics, max_features: int = 500):
        self.intrinsics = intrinsics
        self.max_features = max_features

        # Camera matrix
        self.K = np.array([
            [intrinsics.fx, 0, intrinsics.cx],
            [0, intrinsics.fy, intrinsics.cy],
            [0, 0, 1],
        ])

        # State
        self._pose = np.eye(4)  # current camera pose (world to camera)
        self._prev_features: Optional[np.ndarray] = None
        self._prev_descriptors: Optional[np.ndarray] = None
        self._prev_gray: Optional[np.ndarray] = None
        self._keyframes: List[Dict[str, Any]] = []
        self._map_points: List[np.ndarray] = []
        self._frame_count = 0
        self._tracking_quality = 0.0

    def process_frame(self, rgb_image: np.ndarray,
                      depth_image: Optional[np.ndarray] = None) -> SLAMPose:
        """
        Process a new frame and update pose estimate.

        Args:
            rgb_image: (H, W, 3) uint8 RGB image.
            depth_image: (H, W) float32 depth (improves scale estimation).

        Returns:
            Updated pose estimate.
        """
        gray = np.mean(rgb_image, axis=2).astype(np.uint8)
        self._frame_count += 1

        # Extract features (Harris corners + simple descriptors)
        features = self._extract_features(gray)

        if self._prev_features is None or len(features) < 10:
            self._prev_features = features
            self._prev_gray = gray
            self._tracking_quality = 0.5
            return self._make_slam_pose(len(features))

        # Match features between frames
        matches = self._match_features(gray, features)

        if len(matches) < 8:
            logger.warning(f"SLAM: insufficient matches ({len(matches)}), pose uncertain")
            self._prev_features = features
            self._prev_gray = gray
            self._tracking_quality = max(self._tracking_quality - 0.1, 0.0)
            return self._make_slam_pose(len(matches))

        # Estimate relative motion
        src_pts = self._prev_features[matches[:, 0]]
        dst_pts = features[matches[:, 1]]

        # Essential matrix estimation with RANSAC
        E, inlier_mask = self._estimate_essential_matrix(src_pts, dst_pts)

        if E is not None:
            # Recover rotation and translation from Essential matrix
            R, t = self._decompose_essential(E, src_pts, dst_pts, inlier_mask)

            # Scale from depth if available
            scale = 1.0
            if depth_image is not None:
                scale = self._estimate_scale(depth_image, dst_pts, inlier_mask)

            # Update global pose
            T_relative = np.eye(4)
            T_relative[:3, :3] = R
            T_relative[:3, 3] = t.flatten() * scale
            self._pose = self._pose @ T_relative

            n_inliers = int(np.sum(inlier_mask)) if inlier_mask is not None else len(matches)
            self._tracking_quality = min(n_inliers / self.max_features * 2.0, 1.0)

            # Add keyframe if enough motion or low quality
            if self._should_add_keyframe():
                self._add_keyframe(gray, features, depth_image)

        self._prev_features = features
        self._prev_gray = gray

        return self._make_slam_pose(len(matches))

    def _extract_features(self, gray: np.ndarray) -> np.ndarray:
        """Extract corner features using Harris detector."""
        h, w = gray.shape

        # Compute gradients
        Ix = np.zeros_like(gray, dtype=np.float32)
        Iy = np.zeros_like(gray, dtype=np.float32)
        Ix[:, 1:-1] = (gray[:, 2:].astype(float) - gray[:, :-2].astype(float)) / 2.0
        Iy[1:-1, :] = (gray[2:, :].astype(float) - gray[:-2, :].astype(float)) / 2.0

        # Structure tensor components
        Ixx = Ix * Ix
        Iyy = Iy * Iy
        Ixy = Ix * Iy

        # Box filter (simple Gaussian approximation)
        k = 3
        for arr in [Ixx, Iyy, Ixy]:
            # Row convolution
            tmp = np.cumsum(arr, axis=1)
            arr[:, k:] = tmp[:, k:] - tmp[:, :-k]
            arr[:, :k] = tmp[:, :k]
            # Column convolution
            tmp = np.cumsum(arr, axis=0)
            arr[k:, :] = tmp[k:, :] - tmp[:-k, :]
            arr[:k, :] = tmp[:k, :]

        # Harris response: det(M) - alpha * trace(M)^2
        alpha = 0.04
        det = Ixx * Iyy - Ixy * Ixy
        trace = Ixx + Iyy
        response = det - alpha * trace * trace

        # Non-maximum suppression with grid-based selection
        features = []
        cell_size = max(h, w) // 20
        threshold = np.percentile(response[response > 0], 90) if np.any(response > 0) else 0

        for cy in range(0, h - cell_size, cell_size):
            for cx in range(0, w - cell_size, cell_size):
                cell = response[cy:cy+cell_size, cx:cx+cell_size]
                if cell.max() > threshold:
                    local_y, local_x = np.unravel_index(cell.argmax(), cell.shape)
                    features.append([cx + local_x, cy + local_y])

        features = np.array(features[:self.max_features], dtype=np.float32)
        return features if len(features) > 0 else np.zeros((0, 2), dtype=np.float32)

    def _match_features(self, gray: np.ndarray,
                        features: np.ndarray) -> np.ndarray:
        """Match features between previous and current frame using optical flow."""
        if self._prev_gray is None or len(self._prev_features) == 0 or len(features) == 0:
            return np.zeros((0, 2), dtype=int)

        matches = []
        search_radius = 30  # pixels

        for i, prev_pt in enumerate(self._prev_features):
            px, py = int(prev_pt[0]), int(prev_pt[1])

            # Extract patch around previous feature
            half_patch = 5
            if (py - half_patch < 0 or py + half_patch >= self._prev_gray.shape[0] or
                    px - half_patch < 0 or px + half_patch >= self._prev_gray.shape[1]):
                continue
            patch_prev = self._prev_gray[
                py-half_patch:py+half_patch+1,
                px-half_patch:px+half_patch+1,
            ].astype(float)

            best_score = float('inf')
            best_j = -1

            for j, curr_pt in enumerate(features):
                cx, cy_pt = int(curr_pt[0]), int(curr_pt[1])
                dist = math.sqrt((px - cx)**2 + (py - cy_pt)**2)
                if dist > search_radius:
                    continue

                if (cy_pt - half_patch < 0 or cy_pt + half_patch >= gray.shape[0] or
                        cx - half_patch < 0 or cx + half_patch >= gray.shape[1]):
                    continue

                patch_curr = gray[
                    cy_pt-half_patch:cy_pt+half_patch+1,
                    cx-half_patch:cx+half_patch+1,
                ].astype(float)

                # SSD (Sum of Squared Differences)
                score = np.sum((patch_prev - patch_curr)**2)
                if score < best_score:
                    best_score = score
                    best_j = j

            if best_j >= 0 and best_score < 5000:
                matches.append([i, best_j])

        return np.array(matches, dtype=int) if matches else np.zeros((0, 2), dtype=int)

    def _estimate_essential_matrix(self, src: np.ndarray,
                                    dst: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Estimate essential matrix using 8-point algorithm with RANSAC."""
        n = len(src)
        if n < 8:
            return None, None

        # Normalize points
        src_h = np.column_stack([src, np.ones(n)])
        dst_h = np.column_stack([dst, np.ones(n)])
        src_norm = (np.linalg.inv(self.K) @ src_h.T).T
        dst_norm = (np.linalg.inv(self.K) @ dst_h.T).T

        best_E = None
        best_inliers = None
        best_count = 0
        threshold = 0.01  # normalized coordinates

        rng = np.random.RandomState(int(time.time()) % 2**31)

        for _ in range(200):
            idx = rng.choice(n, 8, replace=False)
            s = src_norm[idx, :2]
            d = dst_norm[idx, :2]

            # Build constraint matrix A
            A = np.zeros((8, 9))
            for k in range(8):
                x1, y1 = s[k]
                x2, y2 = d[k]
                A[k] = [x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1]

            try:
                _, _, Vt = np.linalg.svd(A)
                E = Vt[-1].reshape(3, 3)

                # Enforce rank-2 constraint
                U, S, Vt2 = np.linalg.svd(E)
                S = np.array([1, 1, 0])
                E = U @ np.diag(S) @ Vt2
            except np.linalg.LinAlgError:
                continue

            # Count inliers
            inlier_mask = np.zeros(n, dtype=bool)
            for k in range(n):
                p1 = src_norm[k]
                p2 = dst_norm[k]
                error = abs(p2 @ E @ p1)
                if error < threshold:
                    inlier_mask[k] = True

            count = int(np.sum(inlier_mask))
            if count > best_count:
                best_count = count
                best_E = E
                best_inliers = inlier_mask

        return best_E, best_inliers

    def _decompose_essential(self, E: np.ndarray, src: np.ndarray,
                              dst: np.ndarray,
                              mask: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Decompose essential matrix into R and t."""
        U, _, Vt = np.linalg.svd(E)

        # Ensure proper rotation (det = 1)
        if np.linalg.det(U) < 0:
            U[:, -1] *= -1
        if np.linalg.det(Vt) < 0:
            Vt[-1, :] *= -1

        W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)

        # Four possible solutions
        R1 = U @ W @ Vt
        R2 = U @ W.T @ Vt
        t1 = U[:, 2:3]
        t2 = -U[:, 2:3]

        # Choose solution with most points in front of both cameras
        solutions = [(R1, t1), (R1, t2), (R2, t1), (R2, t2)]
        best_R, best_t = R1, t1
        best_score = 0

        for R, t in solutions:
            score = 0
            for i in range(min(len(src), 20)):
                if mask is not None and not mask[i]:
                    continue
                # Triangulate point
                p_cam = R @ np.array([src[i, 0], src[i, 1], 1.0]) + t.flatten()
                if p_cam[2] > 0:
                    score += 1
            if score > best_score:
                best_score = score
                best_R = R
                best_t = t

        return best_R, best_t

    def _estimate_scale(self, depth_image: np.ndarray,
                        features: np.ndarray,
                        mask: Optional[np.ndarray]) -> float:
        """Estimate motion scale from depth measurements."""
        scales = []
        for i, pt in enumerate(features):
            if mask is not None and not mask[i]:
                continue
            x, y = int(pt[0]), int(pt[1])
            if 0 <= y < depth_image.shape[0] and 0 <= x < depth_image.shape[1]:
                d = depth_image[y, x]
                if 0.1 < d < 10.0:
                    scales.append(d)

        if scales:
            return float(np.median(scales))
        return 1.0

    def _should_add_keyframe(self) -> bool:
        """Decide whether to add current frame as a keyframe."""
        if not self._keyframes:
            return True
        return self._frame_count % 30 == 0  # every 30 frames

    def _add_keyframe(self, gray: np.ndarray, features: np.ndarray,
                      depth: Optional[np.ndarray]) -> None:
        self._keyframes.append({
            "frame_id": self._frame_count,
            "pose": self._pose.copy(),
            "n_features": len(features),
            "timestamp": time.time(),
        })

    def _make_slam_pose(self, n_features: int) -> SLAMPose:
        R = self._pose[:3, :3]
        t = self._pose[:3, 3]

        # Rotation matrix to quaternion
        from scipy.spatial.transform import Rotation as Rot
        quat = Rot.from_matrix(R).as_quat()  # xyzw
        orientation = np.array([quat[3], quat[0], quat[1], quat[2]])  # wxyz

        return SLAMPose(
            position=t.copy(),
            orientation=orientation,
            timestamp=time.time(),
            keyframe_id=len(self._keyframes) - 1,
            num_tracked_features=n_features,
            tracking_quality=self._tracking_quality,
        )

    def get_pose(self) -> np.ndarray:
        """Get current 4x4 pose matrix."""
        return self._pose.copy()

    def get_map_points(self) -> List[np.ndarray]:
        return self._map_points.copy()


# ============================================================================
# Person Tracker
# ============================================================================

class PersonTracker:
    """
    Tracks people across frames with distance estimation for safety.
    Critical for ISO 10218/15066 compliance - the robot must always
    know where humans are relative to its workspace.
    """

    def __init__(self, max_tracks: int = 20, max_age: int = 30):
        self.max_tracks = max_tracks
        self.max_age = max_age
        self._tracks: Dict[int, Dict[str, Any]] = {}
        self._next_id = 0

    def update(self, detections: List[Detection]) -> List[PersonDetection]:
        """
        Update tracks with new person detections.

        Uses Hungarian algorithm approximation for matching,
        creates new tracks for unmatched detections.
        """
        person_dets = [d for d in detections if d.object_class == ObjectClass.PERSON]

        if not person_dets and not self._tracks:
            return []

        # Match detections to existing tracks
        matched, unmatched_dets, unmatched_tracks = self._associate(person_dets)

        # Update matched tracks
        results = []
        for det_idx, track_id in matched:
            det = person_dets[det_idx]
            track = self._tracks[track_id]
            track["bbox"] = det.bbox
            track["position_3d"] = det.position_3d
            track["age"] = 0
            track["hits"] += 1

            # Estimate velocity
            if track["prev_position"] is not None and det.position_3d is not None:
                dt = time.time() - track["last_seen"]
                if dt > 0:
                    track["velocity"] = (det.position_3d - track["prev_position"]) / dt
            track["prev_position"] = det.position_3d.copy() if det.position_3d is not None else None
            track["last_seen"] = time.time()

            results.append(self._track_to_detection(track))

        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            det = person_dets[det_idx]
            track_id = self._next_id
            self._next_id += 1
            self._tracks[track_id] = {
                "id": track_id,
                "bbox": det.bbox,
                "position_3d": det.position_3d,
                "prev_position": None,
                "velocity": np.zeros(3),
                "age": 0,
                "hits": 1,
                "last_seen": time.time(),
            }
            results.append(self._track_to_detection(self._tracks[track_id]))

        # Age out unmatched tracks
        for track_id in unmatched_tracks:
            self._tracks[track_id]["age"] += 1

        # Remove old tracks
        expired = [tid for tid, t in self._tracks.items() if t["age"] > self.max_age]
        for tid in expired:
            del self._tracks[tid]

        return results

    def get_closest_person_distance(self) -> float:
        """Return distance to the closest tracked person (meters)."""
        min_dist = float('inf')
        for track in self._tracks.values():
            if track["position_3d"] is not None:
                dist = float(np.linalg.norm(track["position_3d"]))
                min_dist = min(min_dist, dist)
        return min_dist

    def _associate(self, detections: List[Detection]) -> Tuple[List, List, List]:
        """Simple greedy association based on IoU."""
        matched = []
        unmatched_dets = list(range(len(detections)))
        unmatched_tracks = list(self._tracks.keys())

        if not detections or not self._tracks:
            return matched, unmatched_dets, unmatched_tracks

        # Compute cost matrix (1 - IoU)
        det_indices = list(range(len(detections)))
        track_ids = list(self._tracks.keys())

        for d_idx in det_indices[:]:
            best_iou = 0.3  # minimum IoU threshold
            best_track = -1
            for t_id in track_ids:
                if t_id not in unmatched_tracks:
                    continue
                track = self._tracks[t_id]
                iou = self._compute_iou(detections[d_idx].bbox, track["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_track = t_id

            if best_track >= 0:
                matched.append((d_idx, best_track))
                unmatched_dets.remove(d_idx)
                unmatched_tracks.remove(best_track)

        return matched, unmatched_dets, unmatched_tracks

    def _compute_iou(self, a: BoundingBox, b: BoundingBox) -> float:
        x1 = max(a.x_min, b.x_min)
        y1 = max(a.y_min, b.y_min)
        x2 = min(a.x_max, b.x_max)
        y2 = min(a.y_max, b.y_max)
        if x2 <= x1 or y2 <= y1:
            return 0.0
        intersection = (x2 - x1) * (y2 - y1)
        union = a.area + b.area - intersection
        return intersection / union if union > 0 else 0.0

    def _track_to_detection(self, track: Dict) -> PersonDetection:
        return PersonDetection(
            object_class=ObjectClass.PERSON,
            confidence=min(track["hits"] * 0.1, 1.0),
            bbox=track["bbox"],
            position_3d=track["position_3d"],
            velocity_3d=track["velocity"],
            track_id=track["id"],
        )


# ============================================================================
# Main Vision Pipeline
# ============================================================================

class VisionPipeline:
    """
    Orchestrates the complete vision processing pipeline.

    Processes raw camera data into structured world understanding
    for BrianMind's perceive() method.

    Pipeline:
        RGB + Depth -> Object Detection -> Person Tracking -> 3D Estimation
                    -> Point Cloud -> Floor Detection -> Obstacle Map
                    -> SLAM Pose Update
    """

    def __init__(self, intrinsics: Optional[CameraIntrinsics] = None,
                 config: Optional[Dict[str, Any]] = None):
        config = config or {}

        # Default camera intrinsics (640x480 @ ~60 deg FOV)
        if intrinsics is None:
            intrinsics = CameraIntrinsics(
                fx=525.0, fy=525.0, cx=320.0, cy=240.0,
                width=640, height=480)
        self.intrinsics = intrinsics

        # Subsystems
        self.detector = ObjectDetector(
            confidence_threshold=config.get("detection_confidence", 0.5),
            device=config.get("device", "cpu"),
        )
        self.depth_processor = DepthProcessor(intrinsics)
        self.slam = VisualSLAM(intrinsics)
        self.person_tracker = PersonTracker()

        # State
        self._frame_count = 0
        self._last_process_time = 0.0
        self._fps = 0.0
        self._last_obstacle_map: Optional[np.ndarray] = None
        self._floor_height = 0.0

        logger.info(f"VisionPipeline initialized | "
                    f"resolution={intrinsics.width}x{intrinsics.height} | "
                    f"device={config.get('device', 'cpu')}")

    def process(self, rgb_image: np.ndarray,
                depth_image: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Process a single frame through the complete vision pipeline.

        Args:
            rgb_image: (H, W, 3) uint8 RGB image.
            depth_image: (H, W) float32 depth in meters.

        Returns:
            Dict with keys: detections, people, point_cloud, slam_pose,
            obstacle_map, floor_height, fps
        """
        t_start = time.time()
        self._frame_count += 1

        result = {
            "detections": [],
            "people": [],
            "point_cloud": None,
            "slam_pose": None,
            "obstacle_map": None,
            "floor_height": self._floor_height,
            "closest_person_distance": float('inf'),
            "fps": self._fps,
        }

        # 1. Object detection
        detections = self.detector.detect(rgb_image, depth_image, self.intrinsics)
        result["detections"] = detections

        # 2. Person tracking (critical for safety)
        people = self.person_tracker.update(detections)
        result["people"] = people
        result["closest_person_distance"] = self.person_tracker.get_closest_person_distance()

        # 3. Depth processing (if available)
        if depth_image is not None:
            # Point cloud
            cloud = self.depth_processor.depth_to_pointcloud(depth_image, rgb_image)
            result["point_cloud"] = cloud

            # Floor detection (every 10 frames)
            if self._frame_count % 10 == 0 and len(cloud.points) > 100:
                _, d = self.depth_processor.compute_floor_plane(cloud)
                self._floor_height = -d
                result["floor_height"] = self._floor_height

            # Obstacle map
            obstacle_map = self.depth_processor.compute_obstacle_map(
                cloud, self._floor_height)
            result["obstacle_map"] = obstacle_map
            self._last_obstacle_map = obstacle_map

        # 4. Visual SLAM
        slam_pose = self.slam.process_frame(rgb_image, depth_image)
        result["slam_pose"] = slam_pose

        # FPS tracking
        t_end = time.time()
        dt = t_end - self._last_process_time if self._last_process_time > 0 else 1.0
        self._fps = 1.0 / dt if dt > 0 else 0.0
        self._last_process_time = t_end

        logger.debug(f"Vision frame {self._frame_count} | "
                     f"{len(detections)} objects | {len(people)} people | "
                     f"SLAM quality={slam_pose.tracking_quality:.2f} | "
                     f"{self._fps:.1f} FPS")

        return result

    def get_world_state_updates(self, vision_result: Dict) -> Dict[str, Any]:
        """
        Convert vision pipeline output into WorldState-compatible updates
        for BrianMind's perceive() method.
        """
        detected_objects = []
        for det in vision_result["detections"]:
            if det.object_class != ObjectClass.PERSON:
                obj = {
                    "class": det.object_class.name.lower(),
                    "confidence": det.confidence,
                    "bbox": [det.bbox.x_min, det.bbox.y_min,
                             det.bbox.x_max, det.bbox.y_max],
                    "track_id": det.track_id,
                }
                if det.position_3d is not None:
                    obj["position"] = det.position_3d.tolist()
                detected_objects.append(obj)

        detected_people = []
        for person in vision_result["people"]:
            p = {
                "id": person.track_id,
                "confidence": person.confidence,
                "bbox": [person.bbox.x_min, person.bbox.y_min,
                         person.bbox.x_max, person.bbox.y_max],
            }
            if person.position_3d is not None:
                p["position"] = person.position_3d.tolist()
                p["distance"] = float(np.linalg.norm(person.position_3d))
            if person.velocity_3d is not None:
                p["velocity"] = person.velocity_3d.tolist()
            detected_people.append(p)

        updates = {
            "detected_objects": detected_objects,
            "detected_people": detected_people,
            "occupancy_grid": vision_result.get("obstacle_map"),
        }

        slam_pose = vision_result.get("slam_pose")
        if slam_pose is not None:
            updates["robot_pose"] = {
                "position": slam_pose.position.tolist(),
                "orientation": slam_pose.orientation.tolist(),
                "tracking_quality": slam_pose.tracking_quality,
            }

        return updates

    def get_status(self) -> Dict[str, Any]:
        return {
            "frame_count": self._frame_count,
            "fps": self._fps,
            "floor_height": self._floor_height,
            "slam_keyframes": len(self.slam._keyframes),
            "tracked_people": len(self.person_tracker._tracks),
            "closest_person": self.person_tracker.get_closest_person_distance(),
        }
