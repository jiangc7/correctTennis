import os
import anthropic
import json


def generate_report(stroke_type: str, overall_score: int, dimensions: list[dict]) -> dict:
    """Call Claude API to generate bilingual coaching report. Falls back to rule-based if no API key."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return _fallback_report(overall_score, dimensions)

    client = anthropic.Anthropic()

    dimension_text = "\n".join(
        f"- {d['name_en']} ({d['name_zh']}): {d['score']} pts ({d['status']}) — {d['feedback_en']}"
        for d in dimensions
    )

    prompt = f"""You are a professional tennis coach. Based on the following {stroke_type} stroke analysis data, produce a bilingual coaching report.

Overall score: {overall_score}/100

Dimension breakdown:
{dimension_text}

Output ONLY a JSON object with exactly these keys:
- "summary_en": 2-3 sentence English coaching summary
- "summary_zh": same summary in Chinese
- "improvement_tips_en": list of 3-5 specific actionable tips in English
- "improvement_tips_zh": same tips in Chinese

No extra text, only JSON."""

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}],
    )

    text = message.content[0].text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


def _fallback_report(overall_score: int, dimensions: list[dict]) -> dict:
    poor    = [d for d in dimensions if d["status"] == "poor"]
    warning = [d for d in dimensions if d["status"] == "warning"]
    good    = [d for d in dimensions if d["status"] == "good"]

    if overall_score >= 80:
        summary_en = f"Overall score {overall_score}/100 — your stroke technique is solid. Keep up the training and focus on fine-tuning the details."
        summary_zh = f"综合评分 {overall_score} 分，整体动作较为标准。继续保持当前训练节奏，注重细节打磨。"
    elif overall_score >= 60:
        summary_en = f"Overall score {overall_score}/100 — good fundamentals with room for improvement. Focus on the flagged joint angles."
        summary_zh = f"综合评分 {overall_score} 分，动作基础良好，部分环节需要改进。重点关注标注的关节角度。"
    else:
        summary_en = f"Overall score {overall_score}/100 — there are notable deviations in your technique. Consider working with a coach to correct joint alignment."
        summary_zh = f"综合评分 {overall_score} 分，动作存在明显偏差。建议在教练指导下进行针对性训练，逐步纠正各关节角度。"

    tips_en, tips_zh = [], []
    for d in poor:
        tips_en.append(f"Priority fix — {d['name_en']}: {d['feedback_en']}")
        tips_zh.append(f"重点纠正【{d['name_zh']}】：{d['feedback_zh']}")
    for d in warning:
        tips_en.append(f"Improve — {d['name_en']}: {d['feedback_en']}")
        tips_zh.append(f"注意改善【{d['name_zh']}】：{d['feedback_zh']}")
    if good:
        good_names_en = ", ".join(d["name_en"] for d in good)
        good_names_zh = "、".join(d["name_zh"] for d in good)
        tips_en.append(f"Maintain the good form on: {good_names_en}")
        tips_zh.append(f"保持【{good_names_zh}】的良好状态")
    if not tips_en:
        tips_en = ["Keep recording and comparing your sessions to track progress"]
        tips_zh = ["继续录制视频，对比每次训练的进步"]

    return {
        "summary_en": summary_en,
        "summary_zh": summary_zh,
        "improvement_tips_en": tips_en,
        "improvement_tips_zh": tips_zh,
    }
