import re
import threading
import time
import uuid
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, Response
import cv2
import aiofiles

from app.models.schemas import (AnalysisResponse, StrokeAnalysis, StrokeDimension, CompareInfo,
                                PhaseAnalysis, RescoreRequest)
from app.services.pose_analyzer import analyze_video, process_comparison_pair
from app.services.stroke_scorer import score_stroke
from app.services.ai_report import generate_report
from app.services.phase_scorer import detect_phases, score_phases
from app.core.config import settings, serve_reference_candidates

router = APIRouter()

UPLOAD_DIR = Path(settings.data_dir) / "uploads"
OUTPUT_DIR = Path(settings.data_dir) / "outputs"
REF_DIR    = Path(settings.data_dir) / "references"
for d in (UPLOAD_DIR, OUTPUT_DIR, REF_DIR):
    d.mkdir(parents=True, exist_ok=True)

_tasks: dict[str, AnalysisResponse] = {}
_task_created: dict[str, float] = {}

# Heavy CPU work: run at most N analyses at once (others wait, still shown as "processing")
_job_slots = threading.BoundedSemaphore(max(1, settings.max_concurrent_jobs))
_ref_lock = threading.Lock()


def _first_existing(paths: list[Path]) -> str | None:
    for p in paths:
        if p.exists():
            return str(p)
    return None


# stroke_type -> (list of candidate source paths, label)
REFERENCE_VIDEOS = {
    "serve": (serve_reference_candidates(), settings.ref_serve_label),
}

_ref_cache: dict[str, dict] = {}
# Per-task data needed to re-score phases when the user moves their contact point by hand
_phase_store: dict[str, dict] = {}

STROKE_TYPES = ["forehand", "backhand", "serve", "volley", "smash", "slice"]


def _get_ref_data(stroke_type: str) -> dict | None:
    if stroke_type in _ref_cache:
        return _ref_cache[stroke_type]
    if stroke_type not in REFERENCE_VIDEOS:
        return None
    candidates, label = REFERENCE_VIDEOS[stroke_type]
    if isinstance(candidates, (str, Path)):
        candidates = [Path(candidates)]
    src_path = _first_existing(candidates)
    if not src_path:
        return None
    overlay_out  = str(REF_DIR / f"ref_{stroke_type}_overlay.mp4")
    skeleton_out = str(REF_DIR / f"ref_{stroke_type}_skeleton.mp4")
    with _ref_lock:                       # several requests may ask for the reference at once
        if stroke_type in _ref_cache:
            return _ref_cache[stroke_type]
        return _process_reference(stroke_type, src_path, label, overlay_out, skeleton_out)


def _process_reference(stroke_type, src_path, label, overlay_out, skeleton_out) -> dict | None:
    try:
        info = process_comparison_pair(src_path, overlay_out, skeleton_out)
        _ref_cache[stroke_type] = {
            **info,
            "overlay_path":  overlay_out,
            "skeleton_path": skeleton_out,
            "label": label,
        }
        return _ref_cache[stroke_type]
    except Exception as e:
        print(f"Reference processing failed: {e}")
        return None


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    stroke_type: str = Form("serve"),
):
    if stroke_type not in STROKE_TYPES:
        raise HTTPException(400, f"stroke_type must be one of {STROKE_TYPES}")
    if video.content_type and not video.content_type.startswith(("video/", "application/octet-stream")):
        raise HTTPException(415, "Please upload a video file")
    task_id = str(uuid.uuid4())
    # Never trust the client's filename in a path: keep only a safe basename
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(video.filename or "video.mp4").name)[-80:] or "video.mp4"
    input_path   = UPLOAD_DIR / f"{task_id}_{safe_name}"
    output_path  = OUTPUT_DIR / f"{task_id}_annotated.mp4"
    user_overlay = OUTPUT_DIR / f"{task_id}_cmp_overlay.mp4"
    user_skel    = OUTPUT_DIR / f"{task_id}_cmp_skeleton.mp4"

    # Stream to disk in chunks and stop as soon as the size limit is exceeded
    limit = settings.max_upload_mb * 1024 * 1024
    written = 0
    async with aiofiles.open(input_path, "wb") as f:
        while chunk := await video.read(1024 * 1024):
            written += len(chunk)
            if written > limit:
                break
            await f.write(chunk)
    if written > limit:
        input_path.unlink(missing_ok=True)
        raise HTTPException(413, f"Video is larger than {settings.max_upload_mb} MB")
    if written == 0:
        input_path.unlink(missing_ok=True)
        raise HTTPException(400, "Empty file")

    _tasks[task_id] = AnalysisResponse(task_id=task_id, status="processing")
    _task_created[task_id] = time.time()
    background_tasks.add_task(
        _run_analysis, task_id,
        str(input_path), str(output_path),
        str(user_overlay), str(user_skel), stroke_type
    )
    return _tasks[task_id]


@router.get("/status/{task_id}", response_model=AnalysisResponse)
def get_status(task_id: str):
    task = _tasks.get(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    return task


def _video_response(path: Path):
    if not path.exists():
        raise HTTPException(404, "Video not ready")
    return FileResponse(str(path), media_type="video/mp4", headers={"Accept-Ranges": "bytes"})


@router.api_route("/video/{task_id}", methods=["GET", "HEAD"])
def get_video(task_id: str):
    return _video_response(OUTPUT_DIR / f"{task_id}_annotated.mp4")

@router.api_route("/cmp/user/overlay/{task_id}", methods=["GET", "HEAD"])
def get_user_overlay(task_id: str):
    return _video_response(OUTPUT_DIR / f"{task_id}_cmp_overlay.mp4")

@router.api_route("/cmp/user/skeleton/{task_id}", methods=["GET", "HEAD"])
def get_user_skeleton(task_id: str):
    return _video_response(OUTPUT_DIR / f"{task_id}_cmp_skeleton.mp4")

def _frame_jpeg(video_path: Path, frame: int) -> Response:
    if not video_path.exists():
        raise HTTPException(404, "Video not ready")
    cap = cv2.VideoCapture(str(video_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(frame, n - 1)))
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise HTTPException(404, "Frame not available")
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return Response(buf.tobytes(), media_type="image/jpeg",
                    headers={"Cache-Control": "public, max-age=86400"})


@router.get("/frame/user/{task_id}/{frame}")
def get_user_frame(task_id: str, frame: int):
    # From the comparison clip (face already masked, skeleton drawn)
    return _frame_jpeg(OUTPUT_DIR / f"{task_id}_cmp_overlay.mp4", frame)


@router.get("/frame/ref/{stroke_type}/{frame}")
def get_ref_frame(stroke_type: str, frame: int):
    ref = _get_ref_data(stroke_type)
    if not ref:
        raise HTTPException(404, "No reference video")
    return _frame_jpeg(Path(ref["overlay_path"]), frame)


def _build_phase_analysis(task_id: str, stroke_type: str, user_contact_frame: int) -> PhaseAnalysis | None:
    store = _phase_store.get(task_id)
    ref = _get_ref_data(stroke_type)
    if not store or not ref or not ref.get("features"):
        return None
    up = detect_phases(store["features"], user_contact_frame, store["fps"])
    rp = detect_phases(ref["features"], ref["contact_frame"], ref["fps"])
    res = score_phases(store["features"], up, store["fps"], ref["features"], rp, ref["fps"])
    for ph in res["phases"]:
        ph["user_img_url"] = f"/api/frame/user/{task_id}/{ph['user_frame']}"
        ph["ref_img_url"]  = f"/api/frame/ref/{stroke_type}/{ph['ref_frame']}"
    return PhaseAnalysis(overall_score=res["overall_score"], ref_label=ref["label"], phases=res["phases"])


@router.post("/rescore/{task_id}", response_model=PhaseAnalysis)
def rescore(task_id: str, req: RescoreRequest):
    """Re-run phase scoring with a contact point the user picked by hand (seconds into their clip)."""
    store = _phase_store.get(task_id)
    if not store:
        raise HTTPException(404, "Task not found (results are lost when the backend restarts)")
    frame = int(round(req.user_contact_sec * store["fps"]))
    frame = max(0, min(frame, len(store["features"]) - 1))
    pa = _build_phase_analysis(task_id, store["stroke_type"], frame)
    if pa is None:
        raise HTTPException(404, "No reference for this stroke type")
    task = _tasks.get(task_id)
    if task:
        _tasks[task_id] = task.model_copy(update={"phase_analysis": pa})
    return pa


@router.api_route("/cmp/ref/overlay/{stroke_type}", methods=["GET", "HEAD"])
def get_ref_overlay(stroke_type: str):
    ref = _get_ref_data(stroke_type)
    if not ref:
        raise HTTPException(404, "No reference video")
    return _video_response(Path(ref["overlay_path"]))

@router.api_route("/cmp/ref/skeleton/{stroke_type}", methods=["GET", "HEAD"])
def get_ref_skeleton(stroke_type: str):
    ref = _get_ref_data(stroke_type)
    if not ref:
        raise HTTPException(404, "No reference video")
    return _video_response(Path(ref["skeleton_path"]))


def _run_analysis(task_id: str, input_path: str, output_path: str,
                  user_overlay: str, user_skel: str, stroke_type: str):
    with _job_slots:          # waits here if another analysis is already running
        _run_analysis_locked(task_id, input_path, output_path, user_overlay, user_skel, stroke_type)


def _run_analysis_locked(task_id: str, input_path: str, output_path: str,
                         user_overlay: str, user_skel: str, stroke_type: str):
    try:
        # Annotated full video
        angle_stats = analyze_video(input_path, output_path, blur_face=True)
        angle_stats.pop("_contact_frame", None)
        angle_stats.pop("_fps", None)

        # Comparison pair: wider window than the pro clip, so your video can start first and
        # there is room to move your contact point by hand if auto-detection is off.
        user_info = process_comparison_pair(input_path, user_overlay, user_skel,
                                            window_sec=4.0, blur_face=True)

        # Scoring + report
        scored    = score_stroke(stroke_type, angle_stats)
        ai_result = generate_report(stroke_type, scored["overall_score"], scored["dimensions"])

        dimensions = [
            StrokeDimension(
                name_en=d["name_en"], name_zh=d["name_zh"],
                score=d["score"], status=d["status"],
                feedback_en=d["feedback_en"], feedback_zh=d["feedback_zh"],
            ) for d in scored["dimensions"]
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

        _phase_store[task_id] = {"features": user_info["features"], "fps": user_info["fps"],
                                 "stroke_type": stroke_type}

        compare = None
        phase_analysis = None
        ref = _get_ref_data(stroke_type)
        if ref:
            compare = CompareInfo(
                user_overlay_url=f"/api/cmp/user/overlay/{task_id}",
                user_skeleton_url=f"/api/cmp/user/skeleton/{task_id}",
                ref_overlay_url=f"/api/cmp/ref/overlay/{stroke_type}",
                ref_skeleton_url=f"/api/cmp/ref/skeleton/{stroke_type}",
                user_contact_sec=user_info["contact_frame"] / user_info["fps"],
                ref_contact_sec=ref["contact_frame"] / ref["fps"],
                ref_label=ref["label"],
            )
            phase_analysis = _build_phase_analysis(task_id, stroke_type, user_info["contact_frame"])

        _tasks[task_id] = AnalysisResponse(
            task_id=task_id, status="completed",
            annotated_video_url=f"/api/video/{task_id}",
            analysis=analysis, compare=compare, phase_analysis=phase_analysis,
        )
    except Exception as e:
        import traceback; traceback.print_exc()
        _tasks[task_id] = AnalysisResponse(task_id=task_id, status="failed", error=str(e))


# ---------------------------------------------------------------------------------------------
# Housekeeping
# ---------------------------------------------------------------------------------------------
def cleanup_old_files() -> dict:
    """
    Delete uploads/outputs older than RETENTION_DAYS and forget their in-memory results.
    Rendered reference clips are kept. Safe to call any time; returns what was removed.
    """
    cutoff = time.time() - settings.retention_days * 86400
    removed_files = 0
    for folder in (UPLOAD_DIR, OUTPUT_DIR):
        for f in folder.iterdir():
            try:
                if f.is_file() and f.stat().st_mtime < cutoff:
                    f.unlink()
                    removed_files += 1
            except OSError:
                pass   # in use / already gone — try again next round
    stale = [tid for tid, t0 in list(_task_created.items()) if t0 < cutoff]
    for tid in stale:
        _tasks.pop(tid, None)
        _phase_store.pop(tid, None)
        _task_created.pop(tid, None)
    if removed_files or stale:
        print(f"[cleanup] removed {removed_files} files, {len(stale)} task records")
    return {"files": removed_files, "tasks": len(stale)}
