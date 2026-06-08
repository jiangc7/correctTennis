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


class AnalysisResponse(BaseModel):
    task_id: str
    status: str  # "processing", "completed", "failed"
    annotated_video_url: Optional[str] = None
    analysis: Optional[StrokeAnalysis] = None
    error: Optional[str] = None
