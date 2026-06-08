export interface StrokeDimension {
  name_en: string;
  name_zh: string;
  score: number;
  status: "good" | "warning" | "poor";
  feedback_en: string;
  feedback_zh: string;
}

export interface StrokeAnalysis {
  stroke_type: string;
  overall_score: number;
  dimensions: StrokeDimension[];
  summary_en: string;
  summary_zh: string;
  improvement_tips_en: string[];
  improvement_tips_zh: string[];
}

export interface AnalysisResponse {
  task_id: string;
  status: "processing" | "completed" | "failed";
  annotated_video_url?: string;
  analysis?: StrokeAnalysis;
  error?: string;
}

export const STROKE_TYPES = [
  { value: "forehand", label: "正手 Forehand" },
  { value: "backhand", label: "反手 Backhand" },
  { value: "serve", label: "发球 Serve" },
  { value: "volley", label: "截击 Volley" },
  { value: "smash", label: "高压 Smash" },
  { value: "slice", label: "切削 Slice" },
] as const;
