import uuid
import asyncio
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
import aiofiles

from app.models.schemas import AnalysisResponse, StrokeAnalysis, StrokeDimension
from app.services.pose_analyzer import analyze_video
from app.services.stroke_scorer import score_stroke
from app.services.ai_report import generate_report

router = APIRouter()

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

_tasks: dict[str, AnalysisResponse] = {}

STROKE_TYPES = ["forehand", "backhand", "serve", "volley", "smash", "slice"]


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    stroke_type: str = Form("forehand"),
):
    if stroke_type not in STROKE_TYPES:
        raise HTTPException(400, f"stroke_type must be one of {STROKE_TYPES}")

    task_id = str(uuid.uuid4())
    input_path = UPLOAD_DIR / f"{task_id}_{video.filename}"
    output_path = OUTPUT_DIR / f"{task_id}_annotated.mp4"

    async with aiofiles.open(input_path, "wb") as f:
        content = await video.read()
        await f.write(content)

    _tasks[task_id] = AnalysisResponse(task_id=task_id, status="processing")
    background_tasks.add_task(_run_analysis, task_id, str(input_path), str(output_path), stroke_type)

    return _tasks[task_id]


@router.get("/status/{task_id}", response_model=AnalysisResponse)
def get_status(task_id: str):
    task = _tasks.get(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    return task


@router.api_route("/video/{task_id}", methods=["GET", "HEAD"])
def get_video(task_id: str):
    path = OUTPUT_DIR / f"{task_id}_annotated.mp4"
    if not path.exists():
        raise HTTPException(404, "Video not ready")
    return FileResponse(str(path), media_type="video/mp4", headers={"Accept-Ranges": "bytes"})


def _run_analysis(task_id: str, input_path: str, output_path: str, stroke_type: str):
    try:
        angle_stats = analyze_video(input_path, output_path)
        scored = score_stroke(stroke_type, angle_stats)

        ai_result = generate_report(stroke_type, scored["overall_score"], scored["dimensions"])

        dimensions = [
            StrokeDimension(
                name_en=d["name_en"],
                name_zh=d["name_zh"],
                score=d["score"],
                status=d["status"],
                feedback_en=d["feedback_en"],
                feedback_zh=d["feedback_zh"],
            )
            for d in scored["dimensions"]
        ]

        analysis = StrokeAnalysis(
            stroke_type=stroke_type,
            overall_score=scored["overall_score"],
            dimensions=dimensions,
            summary_en=ai_result.get("summary_en", ""),
            summary_zh=ai_result.get("summary_zh", ""),
            improvement_tips_en=ai_result.get("improvement_tips_en", []),
            improvement_tips_zh=ai_result.get("improvement_tips_zh", []),
        )

        _tasks[task_id] = AnalysisResponse(
            task_id=task_id,
            status="completed",
            annotated_video_url=f"/api/video/{task_id}",
            analysis=analysis,
        )
    except Exception as e:
        _tasks[task_id] = AnalysisResponse(
            task_id=task_id,
            status="failed",
            error=str(e),
        )
