"""
Rule-based scoring for Phase 1 MVP.
Compares mean joint angles against professional reference ranges.
"""

REFERENCE_ANGLES = {
    "forehand": {
        "right_elbow": (120, 160),
        "right_shoulder": (70, 110),
        "right_knee": (140, 170),
        "left_knee": (140, 170),
    },
    "serve": {
        "right_elbow": (130, 170),
        "right_shoulder": (80, 130),
        "right_knee": (150, 180),
    },
    "backhand": {
        "left_elbow": (120, 165),
        "left_shoulder": (60, 110),
        "right_knee": (135, 170),
    },
}

DIMENSION_LABELS = {
    "right_elbow":    {"en": "Right Elbow",    "zh": "右肘关节"},
    "left_elbow":     {"en": "Left Elbow",     "zh": "左肘关节"},
    "right_shoulder": {"en": "Right Shoulder", "zh": "右肩关节"},
    "left_shoulder":  {"en": "Left Shoulder",  "zh": "左肩关节"},
    "right_knee":     {"en": "Right Knee",     "zh": "右膝关节"},
    "left_knee":      {"en": "Left Knee",      "zh": "左膝关节"},
}


def _score_angle(actual: float, lo: float, hi: float) -> tuple[int, str]:
    mid = (lo + hi) / 2
    half_range = (hi - lo) / 2
    deviation = abs(actual - mid)
    if deviation <= half_range:
        return 100, "good"
    elif deviation <= half_range * 1.5:
        score = int(100 - (deviation - half_range) / (half_range * 0.5) * 30)
        return score, "warning"
    else:
        score = max(0, int(100 - (deviation - half_range) / half_range * 60))
        return score, "poor"


def _generate_feedback(joint: str, actual: float, lo: float, hi: float, status: str) -> dict:
    actual_r = round(actual, 1)
    if status == "good":
        return {
            "en": f"Angle {actual_r}° is within the optimal range ({lo}°–{hi}°) ✅",
            "zh": f"角度 {actual_r}° 在标准范围内 ({lo}°–{hi}°) ✅",
        }
    mid = (lo + hi) / 2
    direction_en = "too large" if actual > mid else "too small"
    direction_zh = "偏大" if actual > mid else "偏小"
    severity_en = "slightly" if status == "warning" else "significantly"
    severity_zh = "稍微" if status == "warning" else "明显"
    return {
        "en": f"Angle {actual_r}° is {severity_en} {direction_en}. Aim for {lo}°–{hi}°",
        "zh": f"角度 {actual_r}° {severity_zh}{direction_zh}，建议调整至 {lo}°–{hi}° 范围",
    }


def score_stroke(stroke_type: str, angle_stats: dict) -> dict:
    refs = REFERENCE_ANGLES.get(stroke_type, REFERENCE_ANGLES["forehand"])
    dimensions = []
    scores = []

    for joint, ref_range in refs.items():
        if ref_range is None or joint not in angle_stats:
            continue
        actual = angle_stats[joint]
        lo, hi = ref_range
        score, status = _score_angle(actual, lo, hi)
        scores.append(score)
        feedback = _generate_feedback(joint, actual, lo, hi, status)
        labels = DIMENSION_LABELS.get(joint, {"en": joint, "zh": joint})
        dimensions.append({
            "name_en": labels["en"],
            "name_zh": labels["zh"],
            "score": score,
            "status": status,
            "feedback_en": feedback["en"],
            "feedback_zh": feedback["zh"],
            "actual_angle": round(actual, 1),
            "reference_range": [lo, hi],
        })

    overall = int(sum(scores) / len(scores)) if scores else 50
    return {"overall_score": overall, "dimensions": dimensions}
