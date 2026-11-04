"""
BJJ Live Tempo Analyzer — prototype (with session recording)

Requirements:
    pip install opencv-python mediapipe numpy

Download MediaPipe Pose Landmarker model (Lite):
    https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task

Run:
    python bjj_tempo.py --model pose_landmarker_lite.task --source 0
    python bjj_tempo.py --model pose_landmarker_lite.task --source 0 --reference reference_match.mp4

Keys:
    q = quit
    r = record a 10-second reference from the current camera
    c = clear current reference
    v = pause / resume session recording
    +/- = change tracking/pairing sensitivity

Recording:
    Every session is written to <record-dir>/session_YYYYMMDD_HHMMSS.mp4
    (only the raw camera frame, no overlays). Disable with --no-record.

Notes:
- This is a prototype. Pose detection and tracking may fail under heavy occlusion.
- "Tempo" is an image-space activity score, not a validated physiological intensity metric.
- Video files are written to disk when recording is enabled; tracking history stays in RAM.
"""

import argparse
import datetime
import os
import time
from collections import defaultdict, deque

import cv2
import mediapipe as mp
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# MediaPipe landmark indices
NOSE = 0
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_HIP, RIGHT_HIP = 23, 24

# Landmark visibility threshold
VISIBILITY_MIN = 0.35


def get_pose_points(landmarks):
    """Return Nx2 normalized landmark coordinates and visibility."""
    pts = np.array([[lm.x, lm.y] for lm in landmarks], dtype=np.float32)
    vis = np.array([getattr(lm, "visibility", 1.0) for lm in landmarks],
                   dtype=np.float32)
    return pts, vis


def pose_center_and_scale(points, visibility):
    """Center at hips (fallback shoulders); scale by torso/body bbox."""
    valid = visibility >= VISIBILITY_MIN
    if valid.sum() < 5:
        return None, None

    hips = [LEFT_HIP, RIGHT_HIP]
    shoulders = [LEFT_SHOULDER, RIGHT_SHOULDER]

    if all(valid[i] for i in hips):
        center = points[hips].mean(axis=0)
    elif all(valid[i] for i in shoulders):
        center = points[shoulders].mean(axis=0)
    else:
        center = points[valid].mean(axis=0)

    body_pts = points[valid]
    height = float(np.ptp(body_pts[:, 1]))
    width = float(np.ptp(body_pts[:, 0]))
    scale = max(height, width, 0.05)
    return center, scale


def torso_color(frame, points, visibility):
    """Sample average HSV color around torso; only a weak tracking cue."""
    h, w = frame.shape[:2]
    indices = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]
    valid = [i for i in indices if visibility[i] >= VISIBILITY_MIN]
    if not valid:
        return np.array([0, 0, 0], dtype=np.float32)

    xy = points[valid]
    x1 = max(0, int((xy[:, 0].min() - 0.03) * w))
    y1 = max(0, int((xy[:, 1].min() - 0.03) * h))
    x2 = min(w, int((xy[:, 0].max() + 0.03) * w))
    y2 = min(h, int((xy[:, 1].max() + 0.03) * h))

    if x2 <= x1 or y2 <= y1:
        return np.array([0, 0, 0], dtype=np.float32)

    roi = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # Median is less sensitive to skin pixels and highlights than mean.
    return np.median(hsv.reshape(-1, 3), axis=0).astype(np.float32)


class PoseTracker:
    """Simple greedy tracker using center distance + torso color."""

    def __init__(self, max_distance=0.22, max_lost=12):
        self.max_distance = max_distance
        self.max_lost = max_lost
        self.next_id = 1
        self.tracks = {}

    def update(self, detections, dt=1.0 / 30.0):
        # detections: list of dicts with center, scale, color, points
        track_ids = list(self.tracks.keys())
        candidates = []

        for tid in track_ids:
            tr = self.tracks[tid]
            for di, det in enumerate(detections):
                dist = float(np.linalg.norm(tr["center"] - det["center"]))
                color_dist = float(np.linalg.norm(tr["color"] - det["color"])) / 255.0
                # Position is primary; color is only a small tie-breaker.
                cost = dist + 0.035 * color_dist
                if dist <= self.max_distance:
                    candidates.append((cost, tid, di))

        candidates.sort()
        used_tracks, used_dets = set(), set()
        assignments = {}

        for _, tid, di in candidates:
            if tid in used_tracks or di in used_dets:
                continue
            used_tracks.add(tid)
            used_dets.add(di)
            assignments[di] = tid

        for di, det in enumerate(detections):
            if di not in assignments:
                tid = self.next_id
                self.next_id += 1
                assignments[di] = tid
                self.tracks[tid] = {
                    "center": det["center"].copy(),
                    "scale": det["scale"],
                    "color": det["color"].copy(),
                    "points": det["points"].copy(),
                    "visibility": det["visibility"].copy(),
                    "last_activity": 0.0,
                    "lost": 0,
                }

        for di, det in enumerate(detections):
            tid = assignments[di]
            tr = self.tracks[tid]
            old_points = tr["points"]
            old_center = tr["center"]
            old_scale = tr["scale"]

            # Motion is normalized by body size, making it less dependent on camera distance.
            valid = det["visibility"] >= VISIBILITY_MIN
            old_valid = tr["visibility"] >= VISIBILITY_MIN
            common = valid & old_valid

            if common.sum() >= 8:
                joint_motion = np.linalg.norm(
                    det["points"][common] - old_points[common], axis=1
                ).mean()
            else:
                joint_motion = np.linalg.norm(det["center"] - old_center)

            scale = max((det["scale"] + old_scale) / 2.0, 0.05)
            raw_activity = float(joint_motion / scale / max(dt, 1e-3))

            # Smooth the score to reduce frame-to-frame jitter.
            activity = 0.65 * tr["last_activity"] + 0.35 * raw_activity

            tr.update({
                "center": det["center"].copy(),
                "scale": det["scale"],
                "color": det["color"].copy(),
                "points": det["points"].copy(),
                "visibility": det["visibility"].copy(),
                "last_activity": activity,
                "lost": 0,
            })
            det["id"] = tid
            det["activity"] = activity

        # Tracks not seen this frame are retained briefly for ID stability.
        seen = set(assignments.values())
        for tid in list(self.tracks.keys()):
            if tid not in seen:
                self.tracks[tid]["lost"] += 1
                if self.tracks[tid]["lost"] > self.max_lost:
                    del self.tracks[tid]

        return detections


def detect_poses(frame, landmarker):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)

    detections = []
    for landmarks in result.pose_landmarks:
        points, visibility = get_pose_points(landmarks)
        center, scale = pose_center_and_scale(points, visibility)
        if center is None:
            continue
        detections.append({
            "points": points,
            "visibility": visibility,
            "center": center,
            "scale": scale,
            "color": torso_color(frame, points, visibility),
        })
    return detections


def detect_poses_video(frame, landmarker, timestamp_ms):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)
    detections = []
    for landmarks in result.pose_landmarks:
        points, visibility = get_pose_points(landmarks)
        center, scale = pose_center_and_scale(points, visibility)
        if center is None:
            continue
        detections.append({
            "points": points, "visibility": visibility,
            "center": center, "scale": scale,
            "color": torso_color(frame, points, visibility),
        })
    return detections


def pair_people(detections, max_relative_distance=1.8):
    """Greedy nearest-neighbor pairs. Pairing is approximate in crowded scenes."""
    pairs = []
    used = set()

    # Sort candidate pairs by normalized distance.
    candidates = []
    for i in range(len(detections)):
        for j in range(i + 1, len(detections)):
            a, b = detections[i], detections[j]
            dist = float(np.linalg.norm(a["center"] - b["center"]))
            body_scale = max((a["scale"] + b["scale"]) / 2.0, 0.05)
            relative = dist / body_scale
            if relative <= max_relative_distance:
                candidates.append((relative, i, j))

    candidates.sort()
    for relative, i, j in candidates:
        if i in used or j in used:
            continue
        used.add(i)
        used.add(j)
        pairs.append((detections[i], detections[j], relative))
    return pairs


def load_reference_activity(model_path, video_path, num_poses=30):
    """Estimate mean activity from a reference video using the same pose model."""
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=num_poses,
        min_pose_detection_confidence=0.35,
        min_pose_presence_confidence=0.35,
        min_tracking_confidence=0.35,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        landmarker.close()
        raise RuntimeError(f"Cannot open reference video: {video_path}")

    tracker = PoseTracker()
    activities = []
    timestamp_ms = 0
    frame_index = 0
    fps = max(cap.get(cv2.CAP_PROP_FPS), 1.0)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_index += 1
        timestamp_ms = int(frame_index * 1000 / fps)
        detections = detect_poses_video(frame, landmarker, timestamp_ms)
        detections = tracker.update(detections, dt=1.0 / fps)
        if detections:
            activities.append(float(np.mean([d["activity"] for d in detections])))

    cap.release()
    landmarker.close()
    if not activities:
        raise RuntimeError("No poses found in reference video.")
    return float(np.mean(activities))


class SessionRecorder:
    """Synchronous MP4 session recorder. Codec work happens in C, cheap per frame.

    Writes the raw camera frame (before overlays) to <out_dir>/session_*.mp4.
    Falls back to 'mp4v' if the requested FourCC is unavailable.
    """

    def __init__(self, out_dir, fps, fourcc="mp4v", max_seconds=1800.0):
        self.out_dir = out_dir
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self.fourcc = fourcc
        self.max_seconds = max_seconds  # 0 = disable rotation
        self.writer = None
        self.path = None
        self.started_at = 0.0
        os.makedirs(self.out_dir, exist_ok=True)

    def _open(self, frame):
        h, w = frame.shape[:2]
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(self.out_dir, f"session_{ts}.mp4")

        vw = cv2.VideoWriter(
            self.path,
            cv2.VideoWriter_fourcc(*self.fourcc),
            self.fps, (w, h),
        )
        if not vw.isOpened() and self.fourcc != "mp4v":
            vw = cv2.VideoWriter(
                self.path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.fps, (w, h),
            )
        if not vw.isOpened():
            raise RuntimeError(f"Cannot open VideoWriter: {self.path}")

        self.writer = vw
        self.started_at = time.time()
        print(f"[REC] -> {self.path}")

    def write(self, frame):
        if self.writer is None:
            self._open(frame)
        elif self.max_seconds and (time.time() - self.started_at) > self.max_seconds:
            # Rotate long sessions so we never produce a single huge file.
            self.close()
            self._open(frame)
        self.writer.write(frame)

    def close(self):
        if self.writer is not None:
            self.writer.release()
            print(f"[REC] saved: {self.path}")
        self.writer = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to .task model")
    parser.add_argument("--source", default="0", help="Camera index or video path")
    parser.add_argument("--reference", default=None, help="Reference match video")
    parser.add_argument("--num-poses", type=int, default=30)
    parser.add_argument("--window", type=float, default=8.0,
                        help="Activity smoothing window in seconds")

    # Recording options
    parser.add_argument("--record-dir", default="recordings",
                        help="Folder for .mp4 sessions (empty string = disabled)")
    parser.add_argument("--record-fps", type=float, default=0.0,
                        help="Recording FPS, 0 = use camera FPS")
    parser.add_argument("--record-fourcc", default="mp4v",
                        help="mp4v (safe) | avc1 | XVID")
    parser.add_argument("--record-max-minutes", type=float, default=30.0,
                        help="Rotate file after N minutes, 0 = no rotation")
    parser.add_argument("--no-record", action="store_true",
                        help="Disable session recording entirely")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera/video: {args.source}")

    def make_recorder():
        if args.no_record or not args.record_dir:
            return None
        fps = args.record_fps or cap.get(cv2.CAP_PROP_FPS) or 30.0
        return SessionRecorder(
            args.record_dir, fps, args.record_fourcc,
            max_seconds=args.record_max_minutes * 60.0,
        )

    recorder = make_recorder()

    base_options = python.BaseOptions(model_asset_path=args.model)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=args.num_poses,
        min_pose_detection_confidence=0.35,
        min_pose_presence_confidence=0.35,
        min_tracking_confidence=0.35,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)
    tracker = PoseTracker()

    reference = None
    if args.reference:
        print("Analyzing reference video...")
        reference = load_reference_activity(args.model, args.reference, args.num_poses)
        print(f"Reference activity score: {reference:.5f}")

    activity_history = deque()
    state_durations = defaultdict(float)
    last_state = None
    last_state_change = time.time()
    last_frame_time = time.time()
    frame_index = 0
    start_time = time.time()
    record_until = None
    record_values = []

    pair_distance = 1.8
    print("Running. Press q to quit, r to record 10-second reference, c to clear, "
          "v to pause/resume recording.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Write raw frame to disk BEFORE any drawing/overlays.
            №
            #if recorder is not None:
            #    recorder.write(frame)

            now = time.time()
            dt = max(now - last_frame_time, 1e-3)
            last_frame_time = now
            frame_index += 1

            # MediaPipe VIDEO mode requires monotonically increasing timestamps.
            timestamp_ms = int((now - start_time) * 1000)
            detections = detect_poses_video(frame, landmarker, timestamp_ms)
            detections = tracker.update(detections, dt=dt)
            pairs = pair_people(detections, pair_distance)

            # Use average activity across detected grappling pairs.
            if pairs:
                current_activity = float(np.mean([
                    (a["activity"] + b["activity"]) / 2.0 for a, b, _ in pairs
                ]))
            elif detections:
                current_activity = float(np.mean([d["activity"] for d in detections]))
            else:
                current_activity = 0.0

            activity_history.append((now, current_activity))
            while activity_history and now - activity_history[0][0] > args.window:
                activity_history.popleft()

            smoothed = float(np.mean([v for _, v in activity_history])) if activity_history else 0.0

            if record_until is not None:
                record_values.append(smoothed)
                if now >= record_until:
                    if record_values:
                        reference = float(np.mean(record_values))
                        print(f"New reference activity: {reference:.5f}")
                    record_values.clear()
                    record_until = None

            if reference is None:
                state = "YELLOW"
                ratio = None
            else:
                ratio = smoothed / max(reference, 1e-6) * 100.0
                if 85 <= ratio <= 115:
                    state = "GREEN"
                elif 65 <= ratio < 85 or 115 < ratio <= 135:
                    state = "YELLOW"
                else:
                    state = "RED"

            if last_state is None:
                last_state = state
                last_state_change = now
            elif state != last_state:
                state_durations[last_state] += now - last_state_change
                last_state = state
                last_state_change = now

            # Draw skeletons, IDs, and pair links.
            h, w = frame.shape[:2]
            for det in detections:
                tid = det["id"]
                cx, cy = int(det["center"][0] * w), int(det["center"][1] * h)
                cv2.circle(frame, (cx, cy), 6, (255, 255, 255), -1)
                cv2.putText(frame, f"ID {tid}  A:{det['activity']:.3f}",
                            (cx + 8, cy - 8), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, (255, 255, 255), 1)

            for a, b, _ in pairs:
                ax, ay = int(a["center"][0] * w), int(a["center"][1] * h)
                bx, by = int(b["center"][0] * w), int(b["center"][1] * h)
                cv2.line(frame, (ax, ay), (bx, by), (255, 255, 0), 2)

            colors = {
                "GREEN": (0, 200, 0),
                "YELLOW": (0, 220, 255),
                "RED": (0, 0, 230),
            }
            color = colors[state]
            cv2.rectangle(frame, (w - 230, 15), (w - 15, 115), color, -1)
            cv2.putText(frame, state, (w - 210, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            tempo_text = f"TEMPO: {ratio:.0f}%" if ratio is not None else "Set reference (R)"
            cv2.putText(frame, tempo_text, (w - 210, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            # Show in-memory state durations, including the current state.
            elapsed = now - last_state_change
            durations = dict(state_durations)
            durations[state] = durations.get(state, 0.0) + elapsed
            y = 145
            for name in ("GREEN", "YELLOW", "RED"):
                cv2.putText(frame, f"{name}: {durations.get(name, 0):.1f}s",
                            (w - 220, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, colors[name], 2)
                y += 22

            cv2.putText(frame, f"People: {len(detections)} | Pairs: {len(pairs)}",
                        (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2)

            if record_until is not None:
                cv2.putText(frame, "RECORDING REFERENCE...",
                            (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 255, 255), 2)

            # Recording indicator (raw frames written silently in background).
            if recorder is not None and recorder.writer is not None:
                cv2.circle(frame, (25, 100), 8, (0, 0, 255), -1)
                cv2.putText(frame, "REC", (40, 106),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


              # Write the FINAL overlaid frame (HUD + skeletons + REC dot) to disk.
            if recorder is not None:
                recorder.write(frame)

            cv2.imshow("BJJ Live Tempo Analyzer", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("r"):
                record_values = []
                record_until = time.time() + 10.0
                print("Recording reference for 10 seconds...")
            elif key == ord("c"):
                reference = None
                print("Reference cleared.")
            elif key == ord("v"):
                if recorder is not None:
                    recorder.close()
                    recorder = None
                    print("Recording paused.")
                else:
                    recorder = make_recorder()
                    if recorder is not None:
                        print("Recording resumed.")
            elif key in (ord("+"), ord("=")):
                pair_distance = min(3.0, pair_distance + 0.1)
                print(f"Pairing distance: {pair_distance:.1f}")
            elif key == ord("-"):
                pair_distance = max(0.8, pair_distance - 0.1)
                print(f"Pairing distance: {pair_distance:.1f}")

    finally:
        # Add the final current-state duration; all stats are RAM-only.
        if last_state is not None:
            state_durations[last_state] += time.time() - last_state_change
        print("Session state durations (RAM only):")
        for state in ("GREEN", "YELLOW", "RED"):
            print(f"  {state}: {state_durations.get(state, 0.0):.1f} sec")

        if recorder is not None:
            recorder.close()

        cap.release()
        landmarker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()