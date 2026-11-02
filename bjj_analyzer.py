import argparse
import json
import os
import sys
from pathlib import Path

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

# Connections between landmarks.
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


# Colors for different detected athletes (BGR).
PERSON_COLORS = [
    (0, 255, 0),      # Green
    (0, 165, 255),    # Orange
    (255, 0, 255),    # Purple
    (255, 255, 0),    # Cyan
]


def load_model(model_path: str, max_poses: int):
    """
    Load MediaPipe Pose Landmarker using the modern Tasks API.
    """

    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"MediaPipe model not found: {model_path}\n"
            "Download a Pose Landmarker .task model first."
        )

    base_options = python.BaseOptions(
        model_asset_path=model_path
    )

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
    """
    Convert a MediaPipe landmark into a JSON-friendly dictionary.
    """

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
    """
    Convert a world landmark into a JSON-friendly dictionary.
    World coordinates are expressed in meters.
    """

    return {
        "name": LANDMARK_NAMES[index],
        "x": float(landmark.x),
        "y": float(landmark.y),
        "z": float(landmark.z),
    }


def detect_poses(result):
    """
    Convert MediaPipe detection results into JSON data.

    person_id is the detection index in the current frame.
    It is NOT a persistent identity across the entire video.
    """

    persons = []

    for person_id, landmarks in enumerate(result.pose_landmarks):

        person_data = {
            "person_id": person_id,
            "landmarks": [
                landmark_to_dict(landmark, index)
                for index, landmark in enumerate(landmarks)
            ],
        }

        # World landmarks provide approximate 3D body coordinates.
        if person_id < len(result.pose_world_landmarks):
            person_data["world_landmarks"] = [
                world_landmark_to_dict(landmark, index)
                for index, landmark in enumerate(
                    result.pose_world_landmarks[person_id]
                )
            ]

        persons.append(person_data)

    return persons


def draw_skeleton(frame, persons):
    """
    Draw detected skeletons on the video frame.
    """

    height, width = frame.shape[:2]

    for person_id, landmarks in enumerate(persons):

        color = PERSON_COLORS[
            person_id % len(PERSON_COLORS)
        ]

        points = []

        for landmark in landmarks:

            x = int(landmark.x * width)
            y = int(landmark.y * height)

            points.append((x, y))

        # Draw bones.
        for start_idx, end_idx in POSE_CONNECTIONS:

            if start_idx >= len(points) or end_idx >= len(points):
                continue

            start = points[start_idx]
            end = points[end_idx]

            cv2.line(
                frame,
                start,
                end,
                color,
                thickness=3,
                lineType=cv2.LINE_AA,
            )

        # Draw joints.
        for point in points:

            cv2.circle(
                frame,
                point,
                radius=4,
                color=color,
                thickness=-1,
            )

        # Draw athlete ID near the upper body.
        if len(points) > 12:

            shoulder_x = points[11][0]
            shoulder_y = points[11][1]

            cv2.putText(
                frame,
                f"Person {person_id}",
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
    """
    Analyze a video and export:
    - Video with skeleton overlays.
    - JSON containing pose landmarks.

    sample_rate is the number of pose analyses per second.
    For example, 10 means analyze approximately 10 frames per second.
    """

    if not os.path.isfile(video_path):
        raise FileNotFoundError(
            f"Input video not found: {video_path}"
        )

    if sample_rate <= 0:
        raise ValueError("sample_rate must be greater than 0")

    if max_poses < 1:
        raise ValueError("max_poses must be at least 1")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    duration = total_frames / fps if fps > 0 else 0

    print(f"Video: {video_path}")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps:.2f}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Total frames: {total_frames}")
    print(f"Pose analysis rate: {sample_rate} FPS")

    # Create output directory if necessary.
    Path(output_path).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    Path(json_path).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # MP4 output writer.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():
        cap.release()

        raise RuntimeError(
            f"Could not create output video: {output_path}"
        )

    detector = None

    frames_data = []

    frame_index = 0

    last_analysis_time = -1.0

    # Keep the latest detected poses for frames between analyses.
    last_result = None

    try:

        detector = load_model(
            model_path,
            max_poses
        )

        with tqdm(
            total=total_frames,
            desc="Analyzing BJJ video"
        ) as progress:

            while True:

                ret, frame = cap.read()

                if not ret:
                    break

                timestamp_seconds = frame_index / fps

                timestamp_ms = int(
                    timestamp_seconds * 1000
                )

                # Analyze frames at the requested rate.
                should_analyze = (
                    last_analysis_time < 0
                    or timestamp_seconds - last_analysis_time
                    >= 1.0 / sample_rate
                )

                if should_analyze:

                    rgb_frame = cv2.cvtColor(
                        frame,
                        cv2.COLOR_BGR2RGB
                    )

                    mp_image = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=rgb_frame
                    )

                    try:

                        result = detector.detect_for_video(
                            mp_image,
                            timestamp_ms
                        )

                        last_result = result

                        persons_data = detect_poses(
                            result
                        )

                        frames_data.append({
                            "frame_index": frame_index,
                            "timestamp": timestamp_seconds,
                            "persons": persons_data,
                        })

                    except Exception as error:

                        print(
                            f"\nPose detection error "
                            f"at frame {frame_index}: {error}",
                            file=sys.stderr
                        )

                        last_result = None

                    last_analysis_time = timestamp_seconds

                # Draw the latest detected poses.
                if last_result is not None:

                    frame = draw_skeleton(
                        frame,
                        last_result.pose_landmarks
                    )

                # Display video information.
                cv2.putText(
                    frame,
                    f"Video FPS: {fps:.1f}",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )

                cv2.putText(
                    frame,
                    f"Time: {timestamp_seconds:.2f}s",
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )

                cv2.putText(
                    frame,
                    f"Detected: {len(last_result.pose_landmarks) if last_result else 0}",
                    (20, 105),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
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

    # Export JSON.
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
        },
        "frames": frames_data,
    }

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output_data,
            file,
            indent=2,
            ensure_ascii=False
        )

    print("\nAnalysis complete!")
    print(f"Output video: {output_path}")
    print(f"Output JSON: {json_path}")
    print(f"Analyzed frames: {len(frames_data)}")


def main():

    parser = argparse.ArgumentParser(
        description="Local BJJ pose analyzer using MediaPipe"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input video path"
    )

    parser.add_argument(
        "--output",
        default="output.mp4",
        help="Output video path"
    )

    parser.add_argument(
        "--json",
        default="output.json",
        help="Output JSON path"
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Path to MediaPipe .task model"
    )

    parser.add_argument(
        "--sample-rate",
        type=float,
        default=10.0,
        help="Pose analysis rate in FPS"
    )

    parser.add_argument(
        "--max-poses",
        type=int,
        default=2,
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

        print(
            f"\nERROR: {error}",
            file=sys.stderr
        )

        sys.exit(1)


if __name__ == "__main__":
    main()