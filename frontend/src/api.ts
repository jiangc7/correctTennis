import type { AnalysisResponse } from "./types";

export async function submitVideo(file: File, strokeType: string): Promise<AnalysisResponse> {
  const form = new FormData();
  form.append("video", file);
  form.append("stroke_type", strokeType);
  const res = await fetch("/api/analyze", { method: "POST", body: form });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function pollStatus(taskId: string): Promise<AnalysisResponse> {
  const res = await fetch(`/api/status/${taskId}`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}
