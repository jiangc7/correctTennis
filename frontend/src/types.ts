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

export interface CompareInfo {
  user_overlay_url: string;
  user_skeleton_url: string;
  ref_overlay_url: string;
  ref_skeleton_url: string;
  user_contact_sec: number;
  ref_contact_sec: number;
  ref_label: string;
}

export interface PhaseJoint {
  key: string;
  name_en: string;
  name_zh: string;
  user_angle: number;
  ref_angle: number;
  diff: number;          // you − pro, degrees
  score: number;
  status: "good" | "warning" | "poor";
}

export interface PhaseResult {
  key: "trophy" | "contact" | "follow";
  name_en: string;
  name_zh: string;
  score: number;
  user_frame: number;
  ref_frame: number;
  user_time: number;
  ref_time: number;
  user_img_url: string;
  ref_img_url: string;
  joints: PhaseJoint[];
}

export interface PhaseAnalysis {
  overall_score: number;
  ref_label: string;
  phases: PhaseResult[];
}

export interface AnalysisResponse {
  task_id: string;
  status: "processing" | "completed" | "failed";
  annotated_video_url?: string;
  analysis?: StrokeAnalysis;
  compare?: CompareInfo;
  phase_analysis?: PhaseAnalysis;
  error?: string;
}

export const STROKE_TYPES = [
  { value: "serve",    label: "发球 Serve" },
  { value: "forehand", label: "正手 Forehand" },
  { value: "backhand", label: "反手 Backhand" },
  { value: "volley",   label: "截击 Volley" },
  { value: "smash",    label: "高压 Smash" },
  { value: "slice",    label: "切削 Slice" },
] as const;
