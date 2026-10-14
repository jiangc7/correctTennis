"""
Phase-based serve scoring: find three key moments — trophy position, contact, follow-through —
in both your clip and the pro reference clip, then compare joint angles at the SAME phase.

Frame indices are relative to the comparison clips (the *_cmp_overlay.mp4 files), so the same
index can be used to pull a screenshot from those videos.
"""
import warnings
import numpy as np

PHASES = [
    ("trophy",  "Trophy Position", "奖杯位"),
    ("contact", "Contact",         "击球点"),
    ("follow",  "Follow-through",  "随挥"),
]

# (key, name_en, name_zh) — "hit"/"toss" are resolved to left/right per player
JOINTS = [
    ("hit_elbow",    "Racket-arm elbow",    "持拍手肘角"),
    ("hit_shoulder", "Racket-arm shoulder", "持拍手肩角（上臂抬起）"),
    ("toss_shoulder","Toss-arm shoulder",   "抛球手肩角"),
    ("knee",         "Knee bend (avg)",     "膝关节（两膝平均）"),
    ("trunk_tilt",   "Trunk tilt",          "躯干倾斜"),
]

GOOD_DIFF, WARN_DIFF = 12.0, 25.0   # degrees from the pro's angle


def _nearest_valid(features, idx):
    n = len(features)
    for d in range(0, 6):
        for j in (idx - d, idx + d):
            if 0 <= j < n and features[j] is not None:
                return j
    return None


def _smoothed_peak(features, key) -> int | None:
    ys = np.array([f[key] if f else np.nan for f in features], dtype=float)
    if np.all(np.isnan(ys)):
        return None
    pad = np.pad(ys, 1, mode="edge")
    with warnings.catch_warnings():   # all-NaN windows (no pose) are expected; they stay NaN
        warnings.simplefilter("ignore", RuntimeWarning)
        sm = np.nanmedian(np.stack([pad[:-2], pad[1:-1], pad[2:]]), axis=0)
    return int(np.nanargmin(sm))


def _hitting_side(features) -> str:
    """
    Each wrist reaches its highest point once: the toss arm first, the racket arm at contact.
    So the racket arm is the one whose peak comes LATER (same rule as contact detection,
    and independent of handedness or MediaPipe's left/right labelling).
    """
    lp, rp = _smoothed_peak(features, "l_wrist_y"), _smoothed_peak(features, "r_wrist_y")
    if lp is None or rp is None:
        return "r"
    return "l" if lp > rp else "r"


def _extreme(features, lo, hi, key, highest: bool):
    best, idx = None, None
    for j in range(max(lo, 0), min(hi, len(features))):
        f = features[j]
        if not f:
            continue
        v = f[key]
        if best is None or (v < best if highest else v > best):
            best, idx = v, j
    return idx


def detect_phases(features: list, contact: int, fps: float) -> dict:
    n = len(features)
    contact = int(min(max(contact, 0), n - 1))
    side = _hitting_side(features)
    toss = "l" if side == "r" else "r"

    # Trophy: toss arm fully up (its wrist at its highest) in the 1.2 s before contact
    trophy = _extreme(features, contact - int(1.2 * fps), contact - int(0.1 * fps),
                      f"{toss}_wrist_y", highest=True)
    if trophy is None:
        trophy = max(0, contact - int(0.4 * fps))

    # Follow-through: racket-hand wrist at its lowest within 0.6 s after contact
    follow = _extreme(features, contact + int(0.1 * fps), contact + int(0.6 * fps) + 1,
                      f"{side}_wrist_y", highest=False)
    if follow is None:
        follow = min(n - 1, contact + int(0.4 * fps))

    return {"side": side, "trophy": trophy, "contact": contact, "follow": follow}


def _joint_values(f: dict, side: str) -> dict:
    other = "l" if side == "r" else "r"
    return {
        "hit_elbow":     f[f"{side}_elbow"],
        "hit_shoulder":  f[f"{side}_shoulder"],
        "toss_shoulder": f[f"{other}_shoulder"],
        "knee":          (f["l_knee"] + f["r_knee"]) / 2,
        "trunk_tilt":    f["trunk_tilt"],
    }


def _score(diff: float) -> tuple[int, str]:
    score = 100 if diff <= 8 else max(0, int(round(100 - (diff - 8) * 2)))
    status = "good" if diff <= GOOD_DIFF else "warning" if diff <= WARN_DIFF else "poor"
    return score, status


def score_phases(user_feats, user_phases, user_fps, ref_feats, ref_phases, ref_fps) -> dict:
    phases_out, phase_scores = [], []
    for key, name_en, name_zh in PHASES:
        ui = _nearest_valid(user_feats, user_phases[key])
        ri = _nearest_valid(ref_feats, ref_phases[key])
        joints, scores = [], []
        if ui is not None and ri is not None:
            uv = _joint_values(user_feats[ui], user_phases["side"])
            rv = _joint_values(ref_feats[ri], ref_phases["side"])
            for jk, jen, jzh in JOINTS:
                diff = abs(uv[jk] - rv[jk])
                sc, st = _score(diff)
                scores.append(sc)
                joints.append({"key": jk, "name_en": jen, "name_zh": jzh,
                               "user_angle": round(uv[jk], 1), "ref_angle": round(rv[jk], 1),
                               "diff": round(uv[jk] - rv[jk], 1), "score": sc, "status": st})
        ps = int(round(np.mean(scores))) if scores else 0
        phase_scores.append(ps)
        u_frame = ui if ui is not None else user_phases[key]
        r_frame = ri if ri is not None else ref_phases[key]
        phases_out.append({
            "key": key, "name_en": name_en, "name_zh": name_zh, "score": ps,
            "user_frame": int(u_frame), "ref_frame": int(r_frame),
            "user_time": round(u_frame / user_fps, 3), "ref_time": round(r_frame / ref_fps, 3),
            "joints": joints,
        })
    return {
        "overall_score": int(round(np.mean(phase_scores))) if phase_scores else 0,
        "user_side": user_phases["side"], "ref_side": ref_phases["side"],
        "phases": phases_out,
    }
