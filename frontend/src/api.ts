import type { AnalysisResponse, PhaseAnalysis } from "./types";

/** Turn a FastAPI error ({"detail": "..."}) into a readable message. */
async function errorMessage(res: Response): Promise<string> {
  const text = await res.text();
  try {
    const j = JSON.parse(text);
    if (typeof j.detail === "string") return j.detail;
  } catch { /* not JSON */ }
  return text || `HTTP ${res.status}`;
}

export async function submitVideo(file: File, strokeType: string): Promise<AnalysisResponse> {
  const form = new FormData();
  form.append("video", file);
  form.append("stroke_type", strokeType);
  const res = await fetch("/api/analyze", { method: "POST", body: form });
  if (!res.ok) throw new Error(await errorMessage(res));
  return res.json();
}

export async function pollStatus(taskId: string): Promise<AnalysisResponse> {
  const res = await fetch(`/api/status/${taskId}`);
  if (!res.ok) throw new Error(await errorMessage(res));
  return res.json();
}

/** Re-score the three phases using a contact point picked by hand (seconds into your comparison clip). */
export async function rescorePhases(taskId: string, userContactSec: number): Promise<PhaseAnalysis> {
  const res = await fetch(`/api/rescore/${taskId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_contact_sec: userContactSec }),
  });
  if (!res.ok) throw new Error(await errorMessage(res));
  return res.json();
}
