import json
import os
import sys
from itertools import permutations
from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from AthleteStats import ACTIVITY_WINDOW_SECONDS, MIN_TRACKED_MOVEMENT, AthleteStats
from utils import (
    MAX_DEBUG_SLOTS,
    get_person_color,
    get_shoulders_feature_and_color,
    write_athlete_stats_to_file,
)

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
STANDING_MIN_ANGLE_DEGREES = 60.0
GROUND_MAX_ANGLE_DEGREES = 30.0

REFERENCE_SHORT_SIDE = 1080.0


def ui_scale(frame) -> float:
    """UI scale where 1.0 corresponds approximately to a 1080p short side."""
    height, width = frame.shape[:2]
    return max(0.35, min(height, width) / REFERENCE_SHORT_SIDE)

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

def classify_posture(landmarks, frame_width, frame_height):
    """Classify shoulder-to-hip axis: vertical is standing, horizontal is ground."""
    required = [11, 12, 23, 24]
    if len(landmarks) <= max(required):
        return "unknown", None
    if any(getattr(landmarks[index], "visibility", 1.0) < 0.25 for index in required):
        return "unknown", None

    shoulders = np.array([
        (landmarks[11].x + landmarks[12].x) / 2.0,
        (landmarks[11].y + landmarks[12].y) / 2.0,
    ])
    hips = np.array([
        (landmarks[23].x + landmarks[24].x) / 2.0,
        (landmarks[23].y + landmarks[24].y) / 2.0,
    ])
    # Landmarks are normalized independently by width and height.  Convert to
    # pixel proportions before measuring the physical screen angle.
    axis = (hips - shoulders) * np.array([frame_width, frame_height])
    if np.linalg.norm(axis) == 0:
        return "unknown", None

    # An undirected axis is in [0, 90]: 90 is vertical and 0 horizontal.
    angle = float(np.degrees(np.arctan2(abs(axis[1]), abs(axis[0]))))
    if angle >= STANDING_MIN_ANGLE_DEGREES:  # 90° ± 30°
        return "standing", angle
    if angle <= GROUND_MAX_ANGLE_DEGREES:
        return "ground", angle
    return "unknown", angle

# Color features are defined once in utils.get_shoulders_feature.

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

            # Calculate movement before updating the tracked center.  Coordinates
            # are normalized to the frame dimensions, so speed is expressed in
            # normalized-frame-units per second rather than a physical unit.
            previous_center = track["center"]
            previous_seen = track["last_seen"]
            if (
                detection["center"] is not None
                and previous_center is not None
                and timestamp > previous_seen
            ):
                detection["speed"] = float(
                    np.linalg.norm(detection["center"] - previous_center)
                    / (timestamp - previous_seen)
                )
            else:
                # First observation (or unavailable torso) has no measurable
                # displacement; emit a numeric value for a stable output schema.
                detection["speed"] = 0.0

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

def detect_poses(result, frame, tracker, timestamp, athlete_stats_by_id=None):
    detections = []

    for detection_index, landmarks in enumerate(result.pose_landmarks):
        feature, center, shoulder_color = get_shoulders_feature_and_color(frame, landmarks)

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
            "shoulder_color": shoulder_color,
        }
        detections.append(detection)

    detections = tracker.update(detections, timestamp)

    for detection in detections:
        frame_height, frame_width = frame.shape[:2]
        posture, angle = classify_posture(
            detection["landmarks_raw"], frame_width, frame_height
        )
        detection["posture"] = posture
        detection["torso_angle_degrees"] = angle
        if athlete_stats_by_id is not None:
            athlete_id = detection["person_id"]
            stats = athlete_stats_by_id.setdefault(
                athlete_id, AthleteStats(athlete_id=athlete_id)
            )
            detection["stats"] = stats
            activity_average, is_explosive, is_takedown = stats.add_observation(
                timestamp, detection["speed"], posture
            )
            detection["confirmed_posture"] = stats.get_confirmed_posture()
            detection["raw_posture"] = posture
            detection["posture"] = detection["confirmed_posture"]
            detection["activity_average_3s"] = activity_average
            detection["is_explosive_moment"] = is_explosive
            detection["is_takedown"] = is_takedown

    persons = []
    for detection in detections:
        person_id = detection["person_id"]
        landmarks = detection["landmarks_raw"]

        person_data = {
            "person_id": int(person_id),
            # Runtime-only link: omitted from the serialized JSON below.
            "stats": detection.get("stats"),
            "detection_index": int(detection["detection_index"]),
            "identity_match_cost": detection.get("match_cost"),
            "shoulder_color_bgr": detection.get("shoulder_color"),
            "speed": float(detection["speed"]),
            "posture": detection["posture"],
            "confirmed_posture": detection.get("confirmed_posture", detection["posture"]),
            "torso_angle_degrees": detection["torso_angle_degrees"],
            "landmarks": [
                landmark_to_dict(lm, i)
                for i, lm in enumerate(landmarks)
            ],
        }
        for key in ("activity_average_3s", "is_explosive_moment", "is_takedown"):
            if key in detection:
                person_data[key] = detection[key]

        world_landmarks = detection["world_landmarks_raw"]
        if world_landmarks is not None:
            person_data["world_landmarks"] = [
                world_landmark_to_dict(lm, i)
                for i, lm in enumerate(world_landmarks)
            ]

        persons.append(person_data)

    return persons, detections

def draw_skeleton(frame, detections, frozen=False):
    height, width = frame.shape[:2]
    s = ui_scale(frame)
    line_thickness = max(2, round(3 * s))
    radius = max(3, round(4 * s))
    label_offset = round(20 * s)
    overlay = frame.copy() if frozen else frame

    for detection in detections:
        person_id = detection["person_id"]
        landmarks = detection["landmarks_raw"]

        color = detection.get("shoulder_color") or get_person_color(person_id)

        points = []
        for landmark in landmarks:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            points.append((x, y))

        for start_idx, end_idx in POSE_CONNECTIONS:
            if start_idx >= len(points) or end_idx >= len(points):
                continue
            cv2.line(
                overlay, points[start_idx], points[end_idx], color,
                thickness=line_thickness, lineType=cv2.LINE_AA
            )

        for point in points:
            cv2.circle(overlay, point, radius=radius, color=color, thickness=-1)

        if len(points) > 12:
            shoulder_x = (points[11][0] + points[12][0]) // 2
            shoulder_y = (points[11][1] + points[12][1]) // 2
            posture = detection["posture"]
            cv2.putText(
                overlay,
                f"Athlete {person_id} {posture}",
                (shoulder_x, max(shoulder_y - label_offset, label_offset)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8 * s,
                color,
                max(1, round(2 * s)),
                cv2.LINE_AA,
            )

    if frozen:
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    return frame

def save_result_to_txt(persons_data):
    with open("result.txt", "w", encoding="utf-8") as file:
        for person_data in persons_data:
            file.write(f"Person ID: {person_data['person_id']}\n")
            file.write(f"Detection Index: {person_data['detection_index']}\n")
            file.write(f"Speed: {person_data['speed']:.6f}\n")
            if 'identity_match_cost' in person_data:
                file.write(f"Identity Match Cost: {person_data['identity_match_cost']}\n")
            file.write("Landmarks:\n")
            for landmark in person_data['landmarks']:
                file.write(f"  {landmark['name']}: ({landmark['x']}, {landmark['y']}, {landmark['z']})\n")
            if 'world_landmarks' in person_data:
                file.write("World Landmarks:\n")
                for landmark in person_data['world_landmarks']:
                    file.write(f"  {landmark['name']}: ({landmark['x']}, {landmark['y']}, {landmark['z']})\n")
            file.write("\n")

def build_overall_statistics(athlete_stats_by_id):
    """Return arithmetic means across athletes for the end-of-file summary."""
    stats_list = list(athlete_stats_by_id.values())
    if not stats_list:
        return {
            "athlete_count": 0,
            "speed": 0.0,
            "standup_time_seconds": 0.0,
            "parter_time_seconds": 0.0,
            "unknown_posture_time_seconds": 0.0,
            "explosive_moment_count": 0.0,
        }
    count = len(stats_list)
    return {
        "athlete_count": count,
        "speed": sum(item.get_average_speed() for item in stats_list) / count,
        "standup_time_seconds": sum(item.standuptime for item in stats_list) / count,
        "parter_time_seconds": sum(item.partertime for item in stats_list) / count,
        "unknown_posture_time_seconds": (
            sum(item.unknown_posture_time for item in stats_list) / count
        ),
        "explosive_moment_count": (
            sum(len(item.explosive_moments) for item in stats_list) / count
        ),
    }


def draw_overall_statistics(frame, overall_stats):
    """Draw the live all-athlete summary in a readable panel on the right."""
    lines = [
        "Overall averages across athletes",
        #f"Athlete Count: {overall_stats['athlete_count']}",
        f"  Speed: {overall_stats['speed']:.6f}",
        f"  Standup Time (s): {overall_stats['standup_time_seconds']:.2f}",
        f"  Parter Time (s): {overall_stats['parter_time_seconds']:.2f}",
        f"  Unknown Posture Time (s): {overall_stats['unknown_posture_time_seconds']:.2f}",
        f"  Explosive Moments: {overall_stats['explosive_moment_count']:.2f}",
    ]
    height, width = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    s = ui_scale(frame)
    font_scale = min(0.5 * s, 1.2)
    thickness = max(1, round(2 * s))
    padding = round(6 * s)
    line_height = round(28 * s)
    text_width = max(cv2.getTextSize(line, font, font_scale, thickness)[0][0] for line in lines)
    panel_width = min(text_width + 2 * padding, width - round(10 * s))
    panel_height = line_height * len(lines) + 2 * padding
    margin = round(16 * s)
    x1 = max(margin, width - panel_width - margin)
    y1 = max(margin, height - panel_height - margin)
    x2 = min(width - margin, x1 + panel_width)
    y2 = min(height - margin, y1 + panel_height)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 20, 20), thickness=-1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
    for index, line in enumerate(lines):
        y = y1 + padding + (index + 1) * line_height - round(2 * s)
        cv2.putText(
            frame, line, (x1 + padding, y), font, font_scale,
            (255, 255, 255), thickness, cv2.LINE_AA,
        )


 
def write_overall_stats_to_file(filepath, overall_stats):
    with open(filepath, "a", encoding="utf-8") as file:
        file.write("Overall averages across athletes\n")
        file.write(f"Athlete Count: {overall_stats['athlete_count']}\n")
        file.write(f" Speed: {overall_stats['speed']:.6f}\n")
        file.write(f" Standup Time (s): {overall_stats['standup_time_seconds']:.2f}\n")
        file.write(f" Parter Time (s): {overall_stats['parter_time_seconds']:.2f}\n")
        file.write(f" Unknown Posture Time (s): {overall_stats['unknown_posture_time_seconds']:.2f}\n")
        file.write(f" Explosive Moments: {overall_stats['_explosive_moment_count']:.2f}\n")

def process_video(
    video_path: str,
    output_path: str,
    json_path: str,
    model_path: str,
    sample_rate: float,
    max_poses: int,
):
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
    last_known_detections = {}
    tracker = AthleteTracker()
    athlete_stats_by_id = {}
    overall_stats = build_overall_statistics(athlete_stats_by_id)

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
                            result, frame, tracker, timestamp_seconds,
                            athlete_stats_by_id,
                        )
                        for detection in last_detections:
                            last_known_detections[detection["person_id"]] = detection.copy()

                        last_known_detections = {
                            person_id: detection
                            for person_id, detection in last_known_detections.items()
                            if person_id in tracker.tracks
                        }

                        frames_data.append({
                            "frame_index": frame_index,
                            "timestamp": timestamp_seconds,
                            "persons": [
                                {key: value for key, value in person.items() if key != "stats"}
                                for person in persons_data
                            ],
                        })
                        overall_stats = build_overall_statistics(athlete_stats_by_id)

                    except Exception as error:
                        print(
                            f"\nPose detection error at frame {frame_index}: {error}",
                            file=sys.stderr,
                        )
                        last_detections = []

                    last_analysis_time = timestamp_seconds

                fresh_ids = {detection["person_id"] for detection in last_detections}
                if last_detections:
                    frame = draw_skeleton(frame, last_detections)
                frozen = [detection for person_id, detection in last_known_detections.items()
                          if person_id not in fresh_ids]
                if frozen:
                    frame = draw_skeleton(frame, frozen, frozen=True)

                draw_overall_statistics(frame, overall_stats)

               # cv2.putText(
               #     frame, f"Video FPS: {fps:.1f}", (20, 35),
               #     cv2.FONT_HERSHEY_SIMPLEX, 0.1, (255, 255, 255), 2
               # )
               # cv2.putText(
               #     frame, f"Time: {timestamp_seconds:.2f}s", (20, 70),
               #     cv2.FONT_HERSHEY_SIMPLEX, 0.1, (255, 255, 255), 2
               # )
               # cv2.putText(
               #     frame, f"Detected: {len(last_detections)}", (20, 105),
               #     cv2.FONT_HERSHEY_SIMPLEX, 0.1, (255, 255, 255), 2
               # )

                s = ui_scale(frame)
                slot_top = round(140 * s)
                slot_height = round(135 * s)
                font_scale = 0.6 * s
                thickness = max(1, round(2 * s))
                second_line_offset = round(40 * s)
                for person_id, detection in sorted(last_known_detections.items()):
                    slot = min(person_id, MAX_DEBUG_SLOTS - 1)
                    color = detection.get("shoulder_color") or get_person_color(person_id)
                    y = slot_top + slot * slot_height
                    cv2.putText(
                        frame,
                        f"Athlete {person_id} {detection.get('confirmed_posture', detection['posture'])}",
                        (round(20 * s), y), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, color, thickness, cv2.LINE_AA,
                    )
                    cv2.putText(
                        frame, f"color: {color}",
                        (round(20 * s), y + second_line_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        color,
                        thickness,
                        cv2.LINE_AA,
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
            "color_format": "BGR",
            "movement_statistics": {
                "minimum_tracked_movement": MIN_TRACKED_MOVEMENT,
                "activity_window_seconds": ACTIVITY_WINDOW_SECONDS,
            },
        },
        "athletes": [stats.to_dict() for _, stats in sorted(athlete_stats_by_id.items())],
        "activity_timeline": [
            {"person_id": athlete_id, **moment}
            for athlete_id, stats in sorted(athlete_stats_by_id.items())
            for moment in stats.explosive_moments
        ],
        "overall_statistics": build_overall_statistics(athlete_stats_by_id),
        "frames": frames_data,
    }

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(output_data, file, indent=2, ensure_ascii=False)

    print("\nAnalysis complete!")
    print(f"Output video: {output_path}")
    print(f"Output JSON: {json_path}")
    print(f"Analyzed frames: {len(frames_data)}")

    # Save result to result.txt
    save_result_to_txt([person for frame_data in frames_data for person in frame_data['persons']])

    # Write separate statistics for each athlete.
    for index, stats in enumerate(athlete_stats_by_id.values()):
        write_athlete_stats_to_file(
            "athlete_stats.txt", stats, mode="w" if index == 0 else "a"
        )
    write_overall_stats_to_file(
        "athlete_stats.txt", build_overall_statistics(athlete_stats_by_id)
    )
