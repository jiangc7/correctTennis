import { useState, useCallback, useRef } from "react";
import { submitVideo, pollStatus } from "./api";
import type { AnalysisResponse } from "./types";
import { STROKE_TYPES } from "./types";
import { t, STROKE_TYPES_I18N, type Lang } from "./i18n";
import "./App.css";

export default function App() {
  const [lang, setLang] = useState<Lang>("en");
  const [file, setFile] = useState<File | null>(null);
  const [strokeType, setStrokeType] = useState("forehand");
  const [result, setResult] = useState<AnalysisResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [dragging, setDragging] = useState(false);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const tx = t[lang];

  const handleFile = (f: File) => {
    if (!f.type.startsWith("video/")) { setError(tx.errorVideo); return; }
    setFile(f);
    setError("");
    setResult(null);
  };

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f);
  }, [lang]);

  const onSubmit = async () => {
    if (!file) return;
    setLoading(true);
    setError("");
    setResult(null);
    try {
      const initial = await submitVideo(file, strokeType);
      setResult(initial);
      pollRef.current = setInterval(async () => {
        const status = await pollStatus(initial.task_id);
        setResult(status);
        if (status.status !== "processing") {
          clearInterval(pollRef.current!);
          setLoading(false);
        }
      }, 2000);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <header className="header">
        <span className="logo">🎾</span>
        <div className="header-text">
          <h1>TennisVision AI</h1>
          <p>{tx.appSubtitle}</p>
        </div>
        <button className="lang-btn" onClick={() => setLang(lang === "en" ? "zh" : "en")}>
          {tx.langToggle}
        </button>
      </header>

      <main className="main">
        <div className="upload-card">
          <div
            className={`drop-zone ${dragging ? "dragging" : ""} ${file ? "has-file" : ""}`}
            onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
            onClick={() => document.getElementById("file-input")?.click()}
          >
            <input
              id="file-input"
              type="file"
              accept="video/*"
              hidden
              onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
            />
            {file ? (
              <div className="file-info">
                <span className="file-icon">🎬</span>
                <span>{file.name}</span>
                <span className="file-size">({(file.size / 1024 / 1024).toFixed(1)} MB)</span>
              </div>
            ) : (
              <div className="drop-hint">
                <span className="upload-icon">⬆️</span>
                <p>{tx.dropHint}</p>
                <p className="hint-sub">{tx.dropSub}</p>
              </div>
            )}
          </div>

          <div className="controls">
            <label>{tx.strokeLabel}</label>
            <select value={strokeType} onChange={(e) => setStrokeType(e.target.value)}>
              {STROKE_TYPES.map((s) => (
                <option key={s.value} value={s.value}>
                  {STROKE_TYPES_I18N[s.value]?.[lang] ?? s.value}
                </option>
              ))}
            </select>
            <button className="analyze-btn" onClick={onSubmit} disabled={!file || loading}>
              {loading ? tx.analyzingBtn : tx.analyzeBtn}
            </button>
          </div>

          {error && <div className="error">{error}</div>}
        </div>

        {result && <ResultPanel result={result} lang={lang} />}
      </main>
    </div>
  );
}

function ResultPanel({ result, lang }: { result: AnalysisResponse; lang: Lang }) {
  const tx = t[lang];

  if (result.status === "processing") {
    return (
      <div className="result-card">
        <div className="processing">
          <div className="spinner" />
          <p>{tx.processing}</p>
        </div>
      </div>
    );
  }

  if (result.status === "failed") {
    return <div className="result-card error">{tx.analysisFailed}: {result.error}</div>;
  }

  const { analysis, annotated_video_url } = result;
  if (!analysis) return null;

  const strokeLabel = STROKE_TYPES_I18N[analysis.stroke_type]?.[lang] ?? analysis.stroke_type;
  const summary = lang === "en" ? analysis.summary_en : analysis.summary_zh;
  const tips = lang === "en" ? analysis.improvement_tips_en : analysis.improvement_tips_zh;

  return (
    <div className="result-card">
      <div className="score-header">
        <ScoreRing score={analysis.overall_score} />
        <div className="score-info">
          <h2>{strokeLabel} — {tx.overallScore}</h2>
          <p className="summary">{summary}</p>
        </div>
      </div>

      <div className="dimensions">
        {analysis.dimensions.map((d) => (
          <div key={d.name_en} className={`dimension ${d.status}`}>
            <div className="dim-header">
              <span>{lang === "en" ? d.name_en : d.name_zh}</span>
              <span className="dim-score">{d.score}</span>
            </div>
            <div className="dim-bar">
              <div className="dim-fill" style={{ width: `${d.score}%` }} />
            </div>
            <p className="dim-feedback">{lang === "en" ? d.feedback_en : d.feedback_zh}</p>
          </div>
        ))}
      </div>

      {tips.length > 0 && (
        <div className="tips">
          <h3>{tx.improvementTips}</h3>
          <ul>
            {tips.map((tip, i) => <li key={i}>{tip}</li>)}
          </ul>
        </div>
      )}

      {annotated_video_url && (
        <div className="video-section">
          <h3>{tx.skeletonVideo}</h3>
          <video controls src={annotated_video_url} className="annotated-video" />
        </div>
      )}
    </div>
  );
}

function ScoreRing({ score }: { score: number }) {
  const r = 40, c = 2 * Math.PI * r;
  const color = score >= 80 ? "var(--accent)" : score >= 60 ? "var(--warn)" : "var(--danger)";
  return (
    <svg className="score-ring" viewBox="0 0 100 100">
      <circle cx="50" cy="50" r={r} fill="none" stroke="var(--border)" strokeWidth="8" />
      <circle
        cx="50" cy="50" r={r} fill="none"
        stroke={color} strokeWidth="8"
        strokeDasharray={`${(score / 100) * c} ${c}`}
        strokeLinecap="round"
        transform="rotate(-90 50 50)"
      />
      <text x="50" y="55" textAnchor="middle" fill={color} fontSize="20" fontWeight="bold">{score}</text>
    </svg>
  );
}
