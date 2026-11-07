from pathlib import Path
from itertools import permutations
import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------
# MediaPipe Pose Landmarker
# ---------------------------------------------------------

LANDMARK_NAMES = [
    "NOSE",
    "LEFT_EYE_INNER", "LEFT_EYE", "LEFT_EYE_OUTER",
    "RIGHT_EYE_INNER", "RIGHT_EYE", "RIGHT_EYE_OUTER",
    "LEFT_EAR", "RIGHT_EAR",
    "MOUTH_LEFT", "MOUTH_RIGHT",
    "LEFT_SHOULDER", "RIGHT_SHOULDER",
    "LEFT_ELBOW", "RIGHT_ELBOW",
    "LEFT_WRIST", "RIGHT_WRIST",
    "LEFT_PINKY", "RIGHT_PINKY",
    "LEFT_INDEX", "RIGHT_INDEX",
    "LEFT_THUMB", "RIGHT_THUMB",
    "LEFT_HIP", "RIGHT_HIP",
    "LEFT_KNEE", "RIGHT_KNEE",
    "LEFT_ANKLE", "RIGHT_ANKLE",
    "LEFT_HEEL", "RIGHT_HEEL",
    "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX",
]

POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3),
    (0, 4), (4, 5), (5, 6),
    (9, 10),
    (11, 12),
    (11, 13), (13, 15),
    (15, 17), (15, 19), (15, 21),
    (17, 19),
    (12, 14), (14, 16),
    (16, 18), (16, 20), (16, 22),
    (18, 20),
    (11, 23), (12, 24),
    (23, 24),
    (23, 25), (25, 27),
    (27, 29), (29, 31),
    (27, 31),
    (24, 26), (26, 28),
    (28, 30), (30, 32),
    (28, 32),
]

# Tracker tuning
MAX_TRACK_LOST_SECONDS = 2.5
MAX_MATCH_COST = 0.82
COLOR_WEIGHT = 0.75
POSITION_WEIGHT = 0.25
FEATURE_EMA_ALPHA = 0.25

# Обратная совместимость: то же самое, но под старым именем.
MAX_DEBUG_SLOTS = 10


def _build_person_colors():
    """Create ten evenly spaced, high-saturation BGR colours."""
    hsv = np.array([[[round(index * 179 / MAX_DEBUG_SLOTS), 220, 255]]
                    for index in range(MAX_DEBUG_SLOTS)], dtype=np.uint8)
    return [tuple(map(int, color)) for color in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0]]


PERSON_COLORS = PERSON_COLORS_BGR = _build_person_colors()

def get_person_color(person_id: int):
    """Возвращает цвет в формате BGR для данного person_id."""
    return PERSON_COLORS_BGR[min(person_id, len(PERSON_COLORS_BGR) - 1)]

def load_model(model_path: str, max_poses: int):
    """Load MediaPipe Pose Landmarker using the modern Tasks API."""
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"MediaPipe model not found: {model_path}\n"
            "Download a Pose Landmarker .task model first."
        )

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=max_poses,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.PoseLandmarker.create_from_options(options)

def landmark_to_dict(landmark, index):
    result = {
        "name": LANDMARK_NAMES[index],
        "x": float(landmark.x),
        "y": float(landmark.y),
        "z": float(landmark.z),
    }
    if hasattr(landmark, "visibility"):
        result["visibility"] = float(landmark.visibility)
    if hasattr(landmark, "presence"):
        result["presence"] = float(landmark.presence)
    return result

def world_landmark_to_dict(landmark, index):
    """World coordinates are approximate and expressed in meters."""
    return {
        "name": LANDMARK_NAMES[index],
        "x": float(landmark.x),
        "y": float(landmark.y),
        "z": float(landmark.z),
    }

def get_shoulders_feature_and_color(frame, landmarks):
    """Return the shoulder histogram, center, and representative BGR color."""
    h, w = frame.shape[:2]

    required = [11, 12]
    if len(landmarks) <= max(required):
        return None, None, None

    pts = []
    for idx in required:
        lm = landmarks[idx]
        if getattr(lm, "visibility", 1.0) < 0.25:
            return None, None, None
        pts.append((int(lm.x * w), int(lm.y * h)))

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]

    shoulder_span = max(abs(xs[1] - xs[0]), 1)
    thickness = max(abs(ys[1] - ys[0]), h * 0.04)

    x1 = max(0, int(min(xs) - shoulder_span * 0.18))
    x2 = min(w, int(max(xs) + shoulder_span * 0.18))
    y1 = max(0, int(min(ys) - thickness * 0.08))
    y2 = min(h, int(max(ys) + thickness * 0.08))

    if x2 <= x1 or y2 <= y1:
        return None, None, None

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None, None, None

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Ignore very dark pixels and very low saturation (often mat/white/gray).
    mask = cv2.inRange(hsv, np.array([0, 35, 35]), np.array([179, 255, 255]))
    valid_ratio = float(np.count_nonzero(mask)) / float(mask.size)

    if valid_ratio < 0.08:
        return None, None, None

    hist = cv2.calcHist(
        [hsv], [0, 1], mask, [24, 16], [0, 180, 0, 256]
    )
    cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)

    # Shoulder midpoint, normalized to frame dimensions.
    center = np.array([
        (min(xs) + max(xs)) / (2.0 * w),
        (min(ys) + max(ys)) / (2.0 * h),
    ], dtype=np.float32)

    # Median is robust to small mat/background areas that survive the mask.
    shoulder_color = tuple(map(int, np.median(roi[mask > 0], axis=0)))
    return hist.flatten().astype(np.float32), center, shoulder_color


def get_shoulders_feature(frame, landmarks):
    """Return the shoulder color histogram and midpoint used by the tracker."""
    feature, center, _ = get_shoulders_feature_and_color(frame, landmarks)
    return feature, center

def color_distance(feature_a, feature_b):
    if feature_a is None or feature_b is None:
        return 0.5  # Neutral fallback when color is unavailable.
    a = feature_a.reshape(-1, 1).astype(np.float32)
    b = feature_b.reshape(-1, 1).astype(np.float32)
    return float(np.clip(cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA), 0, 1))

def position_distance(center_a, center_b):
    if center_a is None or center_b is None:
        return 0.5
    # Frame-normalized distance. 0.5 roughly means a large displacement.
    return float(np.clip(np.linalg.norm(center_a - center_b) / 0.5, 0, 1))

class AthleteTracker:
    def __init__(self):
        self.tracks = {}  # persistent_id -> track data
        self.next_id = 0

    def _cost(self, track, detection):
        c_dist = color_distance(track["feature"], detection["feature"])
        p_dist = position_distance(track["center"], detection["center"])
        return COLOR_WEIGHT * c_dist + POSITION_WEIGHT * p_dist

    def update(self, detections, timestamp):
        # Ignore tracks that have been absent for too long.
        active = {
            track_id: track
            for track_id, track in self.tracks.items()
            if timestamp - track["last_seen"] <= MAX_TRACK_LOST_SECONDS
        }
        self.tracks = active

        track_ids = list(self.tracks.keys())
        det_indices = list(range(len(detections)))
        assigned = {}
        used_tracks = set()
        used_dets = set()

        if track_ids and det_indices:
            best_pairs = []
            best_total = float("inf")

            max_pairs = min(len(track_ids), len(det_indices))
            for selected_tracks in permutations(track_ids, max_pairs):
                for selected_dets in permutations(det_indices, max_pairs):
                    pairs = list(zip(selected_tracks, selected_dets))
                    total = sum(
                        self._cost(self.tracks[t_id], detections[d_idx])
                        for t_id, d_idx in pairs
                    )
                    if total < best_total:
                        best_total = total
                        best_pairs = pairs

            for track_id, det_idx in best_pairs:
                cost = self._cost(self.tracks[track_id], detections[det_idx])
                if cost <= MAX_MATCH_COST:
                    assigned[det_idx] = track_id
                    used_tracks.add(track_id)
                    used_dets.add(det_idx)

        for det_idx, detection in enumerate(detections):
            if det_idx in used_dets:
                continue

            track_id = self.next_id
            self.next_id += 1
            assigned[det_idx] = track_id
            self.tracks[track_id] = {
                "feature": detection["feature"],
                "center": detection["center"],
                "last_seen": timestamp,
            }

        for det_idx, track_id in assigned.items():
            detection = detections[det_idx]
            track = self.tracks[track_id]

            new_feature = detection["feature"]
            if new_feature is not None:
                if track["feature"] is None:
                    track["feature"] = new_feature
                else:
                    old = track["feature"]
                    blended = (
                        (1.0 - FEATURE_EMA_ALPHA) * old
                        + FEATURE_EMA_ALPHA * new_feature
                    )
                    total = float(np.sum(blended))
                    if total > 0:
                        blended /= total
                    track["feature"] = blended.astype(np.float32)

            if detection["center"] is not None:
                if track["center"] is None:
                    track["center"] = detection["center"]
                else:
                    track["center"] = (
                        0.65 * track["center"] + 0.35 * detection["center"]
                    ).astype(np.float32)

            track["last_seen"] = timestamp

            detection["person_id"] = track_id
            detection["match_cost"] = (
                float(self._cost(track, detection))
                if track_id in self.tracks else None
            )

        return detections

 
def write_athlete_stats_to_file(filepath, athlete_stats, mode="w") -> None:
    with open(filepath, mode, encoding="utf-8") as file:
        file.write(f"Athlete ID: {athlete_stats.athlete_id}\n")
        file.write(f"Average Speed: {athlete_stats.get_average_speed():.6f}\n")
        file.write(f"Standup Time (s): {athlete_stats.standuptime:.2f}\n")
        file.write(f"Parter Time (s): {athlete_stats.partertime:.2f}\n")
        file.write(f"Unknown Posture Time (s): {athlete_stats.unknown_posture_time:.2f}\n")
        file.write(f"Explosive Moments: {len(athlete_stats.explosive_moments)}\n\n")
