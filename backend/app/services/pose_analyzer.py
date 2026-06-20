import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

POSE_CONNECTIONS = mp_pose.POSE_CONNECTIONS

JOINT_ANGLES = {
    "right_elbow":    (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW,  mp_pose.PoseLandmark.RIGHT_WRIST),
    "left_elbow":     (mp_pose.PoseLandmark.LEFT_SHOULDER,  mp_pose.PoseLandmark.LEFT_ELBOW,   mp_pose.PoseLandmark.LEFT_WRIST),
    "right_knee":     (mp_pose.PoseLandmark.RIGHT_HIP,      mp_pose.PoseLandmark.RIGHT_KNEE,   mp_pose.PoseLandmark.RIGHT_ANKLE),
    "left_knee":      (mp_pose.PoseLandmark.LEFT_HIP,       mp_pose.PoseLandmark.LEFT_KNEE,    mp_pose.PoseLandmark.LEFT_ANKLE),
    "right_shoulder": (mp_pose.PoseLandmark.RIGHT_HIP,      mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
}

# Error rules: (joint, threshold, direction, label_en, label_zh)
# direction "below" = error when angle < threshold, "above" = error when angle > threshold
ERROR_RULES = [
    ("right_elbow",    100, "below", "Elbow too bent",       (0, 80, 255)),
    ("right_elbow",    170, "above", "Elbow overextended",   (0, 80, 255)),
    ("right_shoulder",  60, "below", "Racket prep too low",  (255, 100, 0)),
    ("right_shoulder", 130, "above", "Shoulder over-rotated",(255, 100, 0)),
    ("right_knee",     130, "below", "Knees too bent",       (0, 200, 100)),
    ("right_knee",     175, "above", "Legs too straight",    (0, 200, 100)),
]

JOINT_LANDMARK_IDX = {
    "right_elbow":    mp_pose.PoseLandmark.RIGHT_ELBOW.value,
    "left_elbow":     mp_pose.PoseLandmark.LEFT_ELBOW.value,
    "right_knee":     mp_pose.PoseLandmark.RIGHT_KNEE.value,
    "left_knee":      mp_pose.PoseLandmark.LEFT_KNEE.value,
    "right_shoulder": mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
}


def calculate_angle(a, b, c) -> float:
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def extract_landmarks(results) -> dict:
    if not results.pose_landmarks:
        return {}
    lm = results.pose_landmarks.landmark
    return {i: (lm[i].x, lm[i].y, lm[i].z) for i in range(len(lm))}


def compute_joint_angles(landmarks: dict) -> dict:
    angles = {}
    for name, (a_idx, b_idx, c_idx) in JOINT_ANGLES.items():
        a_val = landmarks.get(a_idx.value)
        b_val = landmarks.get(b_idx.value)
        c_val = landmarks.get(c_idx.value)
        if a_val and b_val and c_val:
            angles[name] = calculate_angle(a_val[:2], b_val[:2], c_val[:2])
    return angles


def draw_annotations(frame, landmarks_px: dict, angles: dict, width: int, height: int):
    """Draw angle values and error labels on frame."""
    # Angle values next to each joint
    for joint, angle in angles.items():
        idx = JOINT_LANDMARK_IDX.get(joint)
        if idx is None or idx not in landmarks_px:
            continue
        x, y = landmarks_px[idx]
        cv2.putText(frame, f"{int(angle)}°", (x + 6, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)

    # Error labels
    errors = _detect_errors(angles, landmarks_px)
    for i, (label, color, px) in enumerate(errors):
        x, y = px
        # Background pill
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        pad = 4
        cv2.rectangle(frame, (x - pad, y - th - pad), (x + tw + pad, y + pad),
                      (20, 20, 20), -1)
        cv2.rectangle(frame, (x - pad, y - th - pad), (x + tw + pad, y + pad),
                      color, 1)
        cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)


def _detect_errors(angles: dict, landmarks_px: dict) -> list[tuple]:
    """Return list of (label, color, pixel_position) for each triggered error rule."""
    errors = []
    for joint, threshold, direction, label, color in ERROR_RULES:
        angle = angles.get(joint)
        if angle is None:
            continue
        triggered = (direction == "below" and angle < threshold) or \
                    (direction == "above" and angle > threshold)
        if not triggered:
            continue
        idx = JOINT_LANDMARK_IDX.get(joint)
        if idx is None or idx not in landmarks_px:
            continue
        x, y = landmarks_px[idx]
        # Offset label above the joint
        errors.append((label, color, (max(x - 60, 4), max(y - 22, 14))))
    return errors


def analyze_video(input_path: str, output_path: str) -> dict:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tmp_path = output_path.replace(".mp4", "_tmp.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out = cv2.VideoWriter(tmp_path, fourcc, fps, (width, height))
    if not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(tmp_path, fourcc, fps, (width, height))

    all_angles: list[dict] = []

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                )

                landmarks = extract_landmarks(results)
                angles = compute_joint_angles(landmarks)
                all_angles.append(angles)

                lm = results.pose_landmarks.landmark
                landmarks_px = {
                    i: (int(lm[i].x * width), int(lm[i].y * height))
                    for i in range(len(lm))
                }
                draw_annotations(frame, landmarks_px, angles, width, height)

            out.write(frame)

    cap.release()
    out.release()

    # Re-encode with ffmpeg for browser H.264 compatibility
    import subprocess, shutil, os
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        ffmpeg_exe = shutil.which("ffmpeg")

    if ffmpeg_exe:
        result = subprocess.run(
            [ffmpeg_exe, "-y", "-i", tmp_path, "-vcodec", "libx264", "-pix_fmt", "yuv420p", output_path],
            capture_output=True
        )
        if result.returncode == 0:
            os.remove(tmp_path)
        else:
            shutil.move(tmp_path, output_path)
    else:
        shutil.move(tmp_path, output_path)

    return _aggregate_angle_stats(all_angles)


def _aggregate_angle_stats(all_angles: list[dict]) -> dict:
    if not all_angles:
        return {}
    keys = all_angles[0].keys()
    return {
        k: float(np.mean([f[k] for f in all_angles if k in f]))
        for k in keys
    }
