from pydantic import BaseModel
from typing import Optional


class StrokeDimension(BaseModel):
    name_en: str
    name_zh: str
    score: int
    status: str  # "good", "warning", "poor"
    feedback_en: str
    feedback_zh: str


class StrokeAnalysis(BaseModel):
    stroke_type: str
    overall_score: int
    dimensions: list[StrokeDimension]
    summary_en: str
    summary_zh: str
    improvement_tips_en: list[str]
    improvement_tips_zh: list[str]


class CompareInfo(BaseModel):
    user_overlay_url: str
    user_skeleton_url: str
    ref_overlay_url: str
    ref_skeleton_url: str
    user_contact_sec: float   # contact point within the trimmed video
    ref_contact_sec: float
    ref_label: str


class PhaseJoint(BaseModel):
    key: str
    name_en: str
    name_zh: str
    user_angle: float
    ref_angle: float
    diff: float          # user - pro, degrees
    score: int
    status: str          # "good", "warning", "poor"


class PhaseResult(BaseModel):
    key: str             # "trophy" | "contact" | "follow"
    name_en: str
    name_zh: str
    score: int
    user_frame: int
    ref_frame: int
    user_time: float
    ref_time: float
    user_img_url: str
    ref_img_url: str
    joints: list[PhaseJoint]


class PhaseAnalysis(BaseModel):
    overall_score: int
    ref_label: str
    phases: list[PhaseResult]


class RescoreRequest(BaseModel):
    user_contact_sec: float


class AnalysisResponse(BaseModel):
    task_id: str
    status: str  # "processing", "completed", "failed"
    annotated_video_url: Optional[str] = None
    analysis: Optional[StrokeAnalysis] = None
    compare: Optional[CompareInfo] = None
    phase_analysis: Optional[PhaseAnalysis] = None
    error: Optional[str] = None
