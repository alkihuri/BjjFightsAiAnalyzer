import argparse
import json
import os
import sys
from pathlib import Path
from itertools import permutations

import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


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

# Stable colors by persistent athlete ID (BGR).
PERSON_COLORS = [
    (0, 255, 0),      # Green
    (0, 165, 255),    # Orange
    (255, 0, 255),    # Purple
    (255, 255, 0),    # Cyan
    (255, 0, 0),      # Blue
    (0, 0, 255),      # Red
]

# Tracker tuning
MAX_TRACK_LOST_SECONDS = 2.5
MAX_MATCH_COST = 0.82
COLOR_WEIGHT = 0.75
POSITION_WEIGHT = 0.25
FEATURE_EMA_ALPHA = 0.25


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


def get_torso_feature(frame, landmarks):
    """
    Build a clothing-color fingerprint from the torso area.

    Uses a normalized 2D HSV histogram (Hue + Saturation), not a single RGB
    pixel. This is more robust to small lighting changes, but can still fail
    when athletes wear similar colors or overlap heavily.
    """
    h, w = frame.shape[:2]

    # MediaPipe indices: shoulders 11/12, hips 23/24.
    required = [11, 12, 23, 24]
    if len(landmarks) <= max(required):
        return None, None

    pts = []
    for idx in required:
        lm = landmarks[idx]
        if getattr(lm, "visibility", 1.0) < 0.25:
            return None, None
        pts.append((int(lm.x * w), int(lm.y * h)))

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]

    torso_width = max(max(xs) - min(xs), int(w * 0.04))
    torso_height = max(max(ys) - min(ys), int(h * 0.04))

    # Expand the shoulder-to-hip box slightly to capture more clothing.
    x1 = max(0, int(min(xs) - torso_width * 0.18))
    x2 = min(w, int(max(xs) + torso_width * 0.18))
    y1 = max(0, int(min(ys) - torso_height * 0.08))
    y2 = min(h, int(max(ys) + torso_height * 0.08))

    if x2 <= x1 or y2 <= y1:
        return None, None

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None, None

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Ignore very dark pixels and very low saturation (often mat/white/gray).
    mask = cv2.inRange(hsv, np.array([0, 35, 35]), np.array([179, 255, 255]))
    valid_ratio = float(np.count_nonzero(mask)) / float(mask.size)

    if valid_ratio < 0.08:
        return None, None

    hist = cv2.calcHist(
        [hsv], [0, 1], mask, [24, 16], [0, 180, 0, 256]
    )
    cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)

    # Approximate torso center, normalized to frame dimensions.
    center = np.array([
        (min(xs) + max(xs)) / (2.0 * w),
        (min(ys) + max(ys)) / (2.0 * h),
    ], dtype=np.float32)

    return hist.flatten().astype(np.float32), center


def color_distance(feature_a, feature_b):
    """Bhattacharyya distance between normalized HSV histograms [0..1]."""
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
    """
    Assign persistent IDs to MediaPipe detections.

    Detection index from MediaPipe is not an identity. This tracker combines
    torso clothing-color histograms and approximate torso position.
    """

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

        # For the usual BJJ case (two athletes), test all possible assignments
        # and select the globally lowest cost instead of greedy matching.
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

        # Unmatched detections start new tracks.
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

        # Update matched track appearance/position using temporal smoothing.
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
                    # Re-normalize histogram after EMA.
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


def detect_poses(result, frame, tracker, timestamp):
    """Convert pose results to detections and assign persistent athlete IDs."""
    detections = []

    for detection_index, landmarks in enumerate(result.pose_landmarks):
        feature, center = get_torso_feature(frame, landmarks)

        detection = {
            "detection_index": detection_index,
            "landmarks_raw": landmarks,
            "world_landmarks_raw": (
                result.pose_world_landmarks[detection_index]
                if detection_index < len(result.pose_world_landmarks)
                else None
            ),
            "feature": feature,
            "center": center,
        }
        detections.append(detection)

    detections = tracker.update(detections, timestamp)

    persons = []
    for detection in detections:
        person_id = detection["person_id"]
        landmarks = detection["landmarks_raw"]

        person_data = {
            "person_id": int(person_id),  # persistent ID, not MediaPipe index
            "detection_index": int(detection["detection_index"]),
            "identity_match_cost": detection.get("match_cost"),
            "landmarks": [
                landmark_to_dict(lm, i)
                for i, lm in enumerate(landmarks)
            ],
        }

        world_landmarks = detection["world_landmarks_raw"]
        if world_landmarks is not None:
            person_data["world_landmarks"] = [
                world_landmark_to_dict(lm, i)
                for i, lm in enumerate(world_landmarks)
            ]

        persons.append(person_data)

    return persons, detections


def draw_skeleton(frame, detections):
    """Draw skeletons using persistent athlete IDs and stable colors."""
    height, width = frame.shape[:2]

    for detection in detections:
        person_id = detection["person_id"]
        landmarks = detection["landmarks_raw"]
        color = PERSON_COLORS[person_id % len(PERSON_COLORS)]

        points = []
        for landmark in landmarks:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            points.append((x, y))

        for start_idx, end_idx in POSE_CONNECTIONS:
            if start_idx >= len(points) or end_idx >= len(points):
                continue
            cv2.line(
                frame, points[start_idx], points[end_idx],
                color, thickness=3, lineType=cv2.LINE_AA
            )

        for point in points:
            cv2.circle(frame, point, radius=4, color=color, thickness=-1)

        if len(points) > 12:
            shoulder_x = (points[11][0] + points[12][0]) // 2
            shoulder_y = (points[11][1] + points[12][1]) // 2
            cv2.putText(
                frame,
                f"Athlete {person_id}",
                (shoulder_x, max(shoulder_y - 20, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
                cv2.LINE_AA,
            )

    return frame


def process_video(
    video_path: str,
    output_path: str,
    json_path: str,
    model_path: str,
    sample_rate: float,
    max_poses: int,
):
    """Analyze video and export skeleton overlay plus landmark JSON."""
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Input video not found: {video_path}")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be greater than 0")
    if max_poses < 1:
        raise ValueError("max_poses must be at least 1")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    print(f"Video: {video_path}")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps:.2f}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Total frames: {total_frames}")
    print(f"Pose analysis rate: {sample_rate} FPS")
    print("Athlete identity: torso HSV color + position tracking")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create output video: {output_path}")

    detector = None
    frames_data = []
    frame_index = 0
    last_analysis_time = -1.0
    last_detections = []
    tracker = AthleteTracker()

    try:
        detector = load_model(model_path, max_poses)

        with tqdm(total=total_frames, desc="Analyzing BJJ video") as progress:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                timestamp_seconds = frame_index / fps
                timestamp_ms = int(timestamp_seconds * 1000)

                should_analyze = (
                    last_analysis_time < 0
                    or timestamp_seconds - last_analysis_time >= 1.0 / sample_rate
                )

                if should_analyze:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=rgb_frame,
                    )

                    try:
                        result = detector.detect_for_video(mp_image, timestamp_ms)
                        persons_data, last_detections = detect_poses(
                            result, frame, tracker, timestamp_seconds
                        )

                        frames_data.append({
                            "frame_index": frame_index,
                            "timestamp": timestamp_seconds,
                            "persons": persons_data,
                        })

                    except Exception as error:
                        print(
                            f"\\nPose detection error at frame {frame_index}: {error}",
                            file=sys.stderr,
                        )
                        last_detections = []

                    last_analysis_time = timestamp_seconds

                # Draw last analyzed poses on intervening video frames.
                if last_detections:
                    frame = draw_skeleton(frame, last_detections)

                cv2.putText(
                    frame, f"Video FPS: {fps:.1f}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
                )
                cv2.putText(
                    frame, f"Time: {timestamp_seconds:.2f}s", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
                )
                cv2.putText(
                    frame, f"Detected: {len(last_detections)}", (20, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
                )

                writer.write(frame)
                frame_index += 1
                progress.update(1)

    finally:
        cap.release()
        writer.release()
        if detector is not None:
            detector.close()
        cv2.destroyAllWindows()

    output_data = {
        "video": {
            "filename": Path(video_path).name,
            "fps": fps,
            "width": width,
            "height": height,
            "duration": duration,
            "total_frames": total_frames,
            "analysis_sample_rate": sample_rate,
            "max_poses": max_poses,
            "identity_tracking": {
                "method": "torso HSV histogram + position",
                "max_lost_seconds": MAX_TRACK_LOST_SECONDS,
                "max_match_cost": MAX_MATCH_COST,
            },
        },
        "frames": frames_data,
    }

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(output_data, file, indent=2, ensure_ascii=False)

    print("\\nAnalysis complete!")
    print(f"Output video: {output_path}")
    print(f"Output JSON: {json_path}")
    print(f"Analyzed frames: {len(frames_data)}")


def main():
    parser = argparse.ArgumentParser(
        description="Local BJJ pose analyzer with persistent athlete IDs"
    )
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--output", default="output.mp4", help="Output video path")
    parser.add_argument("--json", default="output.json", help="Output JSON path")
    parser.add_argument("--model", required=True, help="Path to MediaPipe .task model")
    parser.add_argument(
        "--sample-rate", type=float, default=10.0,
        help="Pose analysis rate in FPS"
    )
    parser.add_argument(
        "--max-poses", type=int, default=2,
        help="Maximum number of detected people"
    )
    args = parser.parse_args()

    try:
        process_video(
            video_path=args.input,
            output_path=args.output,
            json_path=args.json,
            model_path=args.model,
            sample_rate=args.sample_rate,
            max_poses=args.max_poses,
        )
    except Exception as error:
        print(f"\\nERROR: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
