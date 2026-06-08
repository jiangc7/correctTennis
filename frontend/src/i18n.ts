export type Lang = "en" | "zh";

export const t = {
  en: {
    appSubtitle: "AI-Powered Tennis Stroke Analysis",
    dropHint: "Drag & drop a video file here, or click to select",
    dropSub: "Supports MP4, MOV, AVI",
    strokeLabel: "Stroke Type",
    analyzeBtn: "Analyze",
    analyzingBtn: "Analyzing...",
    errorVideo: "Please upload a video file",
    processing: "Analyzing video, please wait...",
    analysisFailed: "Analysis failed",
    overallScore: "Overall Score",
    improvementTips: "Improvement Tips",
    skeletonVideo: "Skeleton Overlay Video",
    langToggle: "中文",
  },
  zh: {
    appSubtitle: "网球动作智能分析系统",
    dropHint: "拖放视频文件至此，或点击选择",
    dropSub: "支持 MP4、MOV、AVI",
    strokeLabel: "击球类型",
    analyzeBtn: "开始分析",
    analyzingBtn: "分析中...",
    errorVideo: "请上传视频文件",
    processing: "正在分析视频，请稍候...",
    analysisFailed: "分析失败",
    overallScore: "综合评分",
    improvementTips: "改进建议",
    skeletonVideo: "骨架叠加视频",
    langToggle: "English",
  },
} as const;

export const STROKE_TYPES_I18N: Record<string, { en: string; zh: string }> = {
  forehand: { en: "Forehand", zh: "正手 Forehand" },
  backhand: { en: "Backhand", zh: "反手 Backhand" },
  serve:    { en: "Serve",    zh: "发球 Serve" },
  volley:   { en: "Volley",   zh: "截击 Volley" },
  smash:    { en: "Smash",    zh: "高压 Smash" },
  slice:    { en: "Slice",    zh: "切削 Slice" },
};
