import cv2
import mediapipe as mp
import numpy as np
import subprocess

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

POSE_CONNECTIONS = mp_pose.POSE_CONNECTIONS

JOINT_ANGLES = {
    "right_elbow":    (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW,   mp_pose.PoseLandmark.RIGHT_WRIST),
    "left_elbow":     (mp_pose.PoseLandmark.LEFT_SHOULDER,  mp_pose.PoseLandmark.LEFT_ELBOW,    mp_pose.PoseLandmark.LEFT_WRIST),
    "right_knee":     (mp_pose.PoseLandmark.RIGHT_HIP,      mp_pose.PoseLandmark.RIGHT_KNEE,    mp_pose.PoseLandmark.RIGHT_ANKLE),
    "left_knee":      (mp_pose.PoseLandmark.LEFT_HIP,       mp_pose.PoseLandmark.LEFT_KNEE,     mp_pose.PoseLandmark.LEFT_ANKLE),
    "right_shoulder": (mp_pose.PoseLandmark.RIGHT_HIP,      mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
}

ERROR_RULES = [
    ("right_elbow",    100, "below", "Elbow too bent",        (0, 80, 255)),
    ("right_elbow",    170, "above", "Elbow overextended",    (0, 80, 255)),
    ("right_shoulder",  60, "below", "Racket prep too low",   (255, 100, 0)),
    ("right_shoulder", 130, "above", "Shoulder over-rotated", (255, 100, 0)),
    ("right_knee",     130, "below", "Knees too bent",        (0, 200, 100)),
    ("right_knee",     175, "above", "Legs too straight",     (0, 200, 100)),
]

JOINT_LANDMARK_IDX = {
    "right_elbow":    mp_pose.PoseLandmark.RIGHT_ELBOW.value,
    "left_elbow":     mp_pose.PoseLandmark.LEFT_ELBOW.value,
    "right_knee":     mp_pose.PoseLandmark.RIGHT_KNEE.value,
    "left_knee":      mp_pose.PoseLandmark.LEFT_KNEE.value,
    "right_shoulder": mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
}

# Skeleton connections and colors for skeleton-only mode
SKELETON_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # arms
    (11, 23), (12, 24), (23, 24),                        # torso
    (23, 25), (25, 27), (24, 26), (26, 28),              # legs
    (0, 11), (0, 12),                                    # head-shoulder
]
BONE_COLOR  = (0, 220, 180)   # teal
JOINT_COLOR = (255, 255, 255)


def calculate_angle(a, b, c) -> float:
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def frame_features(landmarks, width: int, height: int) -> dict:
    """
    Per-frame joint features for phase scoring, computed in PIXEL space (so the video's aspect
    ratio doesn't distort angles). Both body sides are kept; the phase scorer decides which arm
    holds the racket.
    """
    P = lambda i: (landmarks[i].x * width, landmarks[i].y * height)
    L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 11, 12, 13, 14, 15, 16
    L_HIP, R_HIP, L_KN, R_KN, L_AN, R_AN = 23, 24, 25, 26, 27, 28
    sh_mid  = np.mean([P(L_SH), P(R_SH)], axis=0)
    hip_mid = np.mean([P(L_HIP), P(R_HIP)], axis=0)
    trunk = sh_mid - hip_mid
    # 0° = upright, positive = leaning (either way)
    trunk_tilt = float(np.degrees(np.arctan2(abs(trunk[0]), -trunk[1] + 1e-8)))
    return {
        "l_elbow":    calculate_angle(P(L_SH), P(L_EL), P(L_WR)),
        "r_elbow":    calculate_angle(P(R_SH), P(R_EL), P(R_WR)),
        "l_shoulder": calculate_angle(P(L_HIP), P(L_SH), P(L_EL)),
        "r_shoulder": calculate_angle(P(R_HIP), P(R_SH), P(R_EL)),
        "l_knee":     calculate_angle(P(L_HIP), P(L_KN), P(L_AN)),
        "r_knee":     calculate_angle(P(R_HIP), P(R_KN), P(R_AN)),
        "trunk_tilt": trunk_tilt,
        "l_wrist_y":  landmarks[L_WR].y,
        "r_wrist_y":  landmarks[R_WR].y,
    }


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


def _detect_serve_contact(left_y: list[float], right_y: list[float]) -> int:
    """
    Serve contact estimate that doesn't depend on handedness or MediaPipe's left/right labels:
    each wrist reaches its highest point once — the toss arm first, the hitting arm at contact —
    so take the LATER of the two (lightly median-smoothed) peaks.
    """
    if not left_y or not right_y:
        return 0
    def peak(series):
        a = np.asarray(series, dtype=float)
        if len(a) >= 3:
            pad = np.pad(a, 1, mode="edge")
            a = np.median(np.stack([pad[:-2], pad[1:-1], pad[2:]]), axis=0)
        return int(np.argmin(a))
    return max(peak(left_y), peak(right_y))


def _detect_contact_frame(wrist_y_series: list[float]) -> int:
    """Return frame index where right wrist is highest (min Y = highest in image)."""
    if not wrist_y_series:
        return 0
    return int(np.argmin(wrist_y_series))


class FaceMasker:
    """
    Pixelates the player's face using MediaPipe's face landmarks (0-10: nose, eyes, ears, mouth).
    Keeps the last box for a short while when pose detection drops out, so the face never flashes.
    """
    HOLD_FRAMES = 15

    def __init__(self, width: int, height: int):
        self.w, self.h = width, height
        self.box = None
        self.missed = 0

    def update(self, landmarks):
        if landmarks is None:
            self.missed += 1
            if self.missed > self.HOLD_FRAMES:
                self.box = None
            return
        pts = np.array([(landmarks[i].x * self.w, landmarks[i].y * self.h) for i in range(11)])
        sh  = np.array([(landmarks[i].x * self.w, landmarks[i].y * self.h) for i in (11, 12)])
        span = float(np.max(np.ptp(pts, axis=0)))
        shoulder_w = float(np.linalg.norm(sh[0] - sh[1]))
        size = max(span * 1.9, shoulder_w * 0.75, 0.04 * max(self.w, self.h))
        cx, cy = pts.mean(axis=0)
        cy -= size * 0.08   # cover the forehead / hair line too
        self.box = (cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2)
        self.missed = 0

    def apply(self, frame):
        if self.box is None:
            return frame
        x0, y0, x1, y1 = (int(round(v)) for v in self.box)
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, self.w), min(y1, self.h)
        if x1 - x0 < 2 or y1 - y0 < 2:
            return frame
        roi = frame[y0:y1, x0:x1]
        blocks = 6
        small = cv2.resize(roi, (blocks, max(1, int(blocks * roi.shape[0] / roi.shape[1]))),
                           interpolation=cv2.INTER_LINEAR)
        frame[y0:y1, x0:x1] = cv2.resize(small, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)
        return frame


def _without_face(pose_landmarks):
    """Copy of the landmark list with face points (0-10) hidden, so no dots are drawn over a masked face."""
    from mediapipe.framework.formats import landmark_pb2
    out = landmark_pb2.NormalizedLandmarkList()
    out.CopyFrom(pose_landmarks)
    for i in range(11):
        out.landmark[i].visibility = 0.0
    return out


def draw_annotations(frame, landmarks_px: dict, angles: dict, width: int, height: int):
    for joint, angle in angles.items():
        idx = JOINT_LANDMARK_IDX.get(joint)
        if idx is None or idx not in landmarks_px:
            continue
        x, y = landmarks_px[idx]
        # OpenCV's built-in fonts have no "°" glyph, so draw the number only
        cv2.putText(frame, f"{int(angle)}", (x + 6, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)

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
        lx, ly = max(x - 60, 4), max(y - 22, 14)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        cv2.rectangle(frame, (lx - 4, ly - th - 4), (lx + tw + 4, ly + 4), (20, 20, 20), -1)
        cv2.rectangle(frame, (lx - 4, ly - th - 4), (lx + tw + 4, ly + 4), color, 1)
        cv2.putText(frame, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)


def _draw_skeleton_only(frame_blank, landmarks_px: dict, label: str = ""):
    """Draw skeleton on a dark frame (no original video background)."""
    for a, b in SKELETON_CONNECTIONS:
        if a in landmarks_px and b in landmarks_px:
            cv2.line(frame_blank, landmarks_px[a], landmarks_px[b], BONE_COLOR, 2, cv2.LINE_AA)
    for idx, (x, y) in landmarks_px.items():
        if idx in [lm.value for lm in mp_pose.PoseLandmark]:
            cv2.circle(frame_blank, (x, y), 4, JOINT_COLOR, -1, cv2.LINE_AA)
    if label:
        cv2.putText(frame_blank, label, (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA)


def _get_ffmpeg():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        import shutil
        return shutil.which("ffmpeg")


def _normalize_rotation(input_path: str) -> str:
    """
    Re-encode with ffmpeg so rotation metadata is baked into pixel data.
    OpenCV ignores rotation metadata, so we must correct it before processing.
    Returns normalized temp path, or original if ffmpeg unavailable.
    """
    ffmpeg_exe = _get_ffmpeg()
    if not ffmpeg_exe:
        return input_path
    import tempfile, os
    tmp = tempfile.mktemp(suffix=".mp4")
    # ffmpeg auto-applies rotation metadata when decoding by default
    result = subprocess.run(
        [ffmpeg_exe, "-y", "-i", input_path,
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", tmp],
        capture_output=True
    )
    if result.returncode == 0:
        return tmp
    return input_path


def _ffmpeg_encode(tmp_path: str, output_path: str):
    import shutil, os
    ffmpeg_exe = _get_ffmpeg()
    if ffmpeg_exe:
        r = subprocess.run(
            [ffmpeg_exe, "-y", "-i", tmp_path, "-vcodec", "libx264", "-pix_fmt", "yuv420p", output_path],
            capture_output=True
        )
        if r.returncode == 0:
            os.remove(tmp_path)
        else:
            shutil.move(tmp_path, output_path)
    else:
        shutil.move(tmp_path, output_path)


def analyze_video(input_path: str, output_path: str, blur_face: bool = False) -> dict:
    """Process video: overlay skeleton + error annotations on original frames."""
    input_path = _normalize_rotation(input_path)
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tmp_path = output_path.replace(".mp4", "_tmp.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out = cv2.VideoWriter(tmp_path, fourcc, fps, (width, height))
    if not out.isOpened():
        out = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    all_angles: list[dict] = []
    wrist_y_series: list[float] = []
    masker = FaceMasker(width, height) if blur_face else None

    with mp_pose.Pose(static_image_mode=False, model_complexity=1,
                      smooth_landmarks=True,
                      min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            if masker:
                masker.update(results.pose_landmarks.landmark if results.pose_landmarks else None)
                masker.apply(frame)
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, _without_face(results.pose_landmarks) if masker else results.pose_landmarks,
                                          POSE_CONNECTIONS,
                                          landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                landmarks = extract_landmarks(results)
                angles = compute_joint_angles(landmarks)
                all_angles.append(angles)
                lm = results.pose_landmarks.landmark
                landmarks_px = {i: (int(lm[i].x * width), int(lm[i].y * height)) for i in range(len(lm))}
                draw_annotations(frame, landmarks_px, angles, width, height)
                rw = lm[mp_pose.PoseLandmark.RIGHT_WRIST.value]
                wrist_y_series.append(rw.y)
            else:
                wrist_y_series.append(1.0)
            out.write(frame)

    cap.release()
    out.release()
    _ffmpeg_encode(tmp_path, output_path)

    contact_frame = _detect_contact_frame(wrist_y_series)
    stats = _aggregate_angle_stats(all_angles)
    stats["_contact_frame"] = float(contact_frame)
    stats["_fps"] = float(fps)
    return stats


def process_overlay_video(input_path: str, output_path: str) -> dict:
    """Skeleton overlaid on original video (full length). Returns contact_frame and fps."""
    input_path = _normalize_rotation(input_path)
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open: {input_path}")
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tmp_path = output_path.replace(".mp4", "_tmp.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out = cv2.VideoWriter(tmp_path, fourcc, fps, (width, height))
    if not out.isOpened():
        out = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    wrist_y_series: list[float] = []
    with mp_pose.Pose(static_image_mode=False, model_complexity=1, smooth_landmarks=True,
                      min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                lm = results.pose_landmarks.landmark
                wrist_y_series.append(lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].y)
            else:
                wrist_y_series.append(1.0)
            out.write(frame)
    cap.release()
    out.release()
    _ffmpeg_encode(tmp_path, output_path)
    return {"contact_frame": _detect_contact_frame(wrist_y_series), "fps": float(fps)}


WINDOW_SEC = 2.0   # seconds before and after contact point


def process_comparison_pair(input_path: str, overlay_out: str, skeleton_out: str,
                            window_sec: float = WINDOW_SEC, blur_face: bool = False) -> dict:
    """
    Two-pass processing:
    Pass 1 — detect contact frame.
    Pass 2 — write WINDOW_SEC before/after contact frame to two files:
              overlay_out  = skeleton on original video
              skeleton_out = skeleton on black background
    Both outputs are the same duration and synchronized at contact.
    Returns: contact_frame, fps, window_start_frame, total_output_frames.
    """
    # --- Pass 1: detect contact frame ---
    input_path = _normalize_rotation(input_path)
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open: {input_path}")
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    wrist_y: list[float] = []
    left_y:  list[float] = []
    with mp_pose.Pose(static_image_mode=False, model_complexity=1, smooth_landmarks=True,
                      min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                wrist_y.append(lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].y)
                left_y.append(lm[mp_pose.PoseLandmark.LEFT_WRIST.value].y)
            else:
                wrist_y.append(1.0)
                left_y.append(1.0)
    cap.release()

    contact_frame  = _detect_serve_contact(left_y, wrist_y)
    window_frames  = int(window_sec * fps)
    start_frame    = max(0, contact_frame - window_frames)
    end_frame      = min(len(wrist_y) - 1, contact_frame + window_frames)
    total_frames   = end_frame - start_frame + 1

    # --- Pass 2: render trimmed window ---
    cap = cv2.VideoCapture(input_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    def _make_writer(path: str):
        tmp = path.replace(".mp4", "_tmp.mp4")
        w = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"avc1"), fps, (width, height))
        if not w.isOpened():
            w = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        return w, tmp

    ov_out, ov_tmp   = _make_writer(overlay_out)
    sk_out, sk_tmp   = _make_writer(skeleton_out)

    BG_COLOR = (15, 15, 25)
    masker = FaceMasker(width, height) if blur_face else None
    features: list[dict | None] = []   # one entry per output frame (None = no pose found)

    with mp_pose.Pose(static_image_mode=False, model_complexity=1, smooth_landmarks=True,
                      min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        for fi in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            canvas = np.full((height, width, 3), BG_COLOR, dtype=np.uint8)
            features.append(frame_features(results.pose_landmarks.landmark, width, height)
                            if results.pose_landmarks else None)
            if masker:
                masker.update(results.pose_landmarks.landmark if results.pose_landmarks else None)
                masker.apply(frame)

            if results.pose_landmarks:
                # Overlay version: draw on original frame
                mp_drawing.draw_landmarks(frame, _without_face(results.pose_landmarks) if masker else results.pose_landmarks,
                    POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                # Skeleton-only: draw on black canvas
                lm = results.pose_landmarks.landmark
                lm_px = {i: (int(lm[i].x * width), int(lm[i].y * height)) for i in range(len(lm))}
                for a, b in SKELETON_CONNECTIONS:
                    if a in lm_px and b in lm_px:
                        cv2.line(canvas, lm_px[a], lm_px[b], BONE_COLOR, 2, cv2.LINE_AA)
                for idx, (x, y) in lm_px.items():
                    cv2.circle(canvas, (x, y), 4, JOINT_COLOR, -1, cv2.LINE_AA)

            ov_out.write(frame)
            sk_out.write(canvas)

    cap.release()
    ov_out.release()
    sk_out.release()
    _ffmpeg_encode(ov_tmp, overlay_out)
    _ffmpeg_encode(sk_tmp, skeleton_out)

    return {
        "contact_frame": contact_frame - start_frame,  # relative to trimmed video
        "fps": float(fps),
        "window_start": start_frame,
        "total_frames": total_frames,
        "features": features,
    }


def _aggregate_angle_stats(all_angles: list[dict]) -> dict:
    if not all_angles:
        return {}
    keys = all_angles[0].keys()
    return {k: float(np.mean([f[k] for f in all_angles if k in f])) for k in keys}
