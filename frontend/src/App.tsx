import { useState, useCallback, useRef, useEffect } from "react";
import { submitVideo, pollStatus, rescorePhases } from "./api";
import type { AnalysisResponse, CompareInfo, PhaseAnalysis } from "./types";
import { STROKE_TYPES } from "./types";
import { t, STROKE_TYPES_I18N, type Lang } from "./i18n";
import "./App.css";

export default function App() {
  const [lang, setLang] = useState<Lang>("en");
  const [file, setFile] = useState<File | null>(null);
  const [strokeType, setStrokeType] = useState("serve");
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
  const [tab, setTab] = useState<"compare" | "overlay">("compare");
  const [phase, setPhase] = useState<PhaseAnalysis | undefined>(result.phase_analysis);
  const [rescoring, setRescoring] = useState(false);
  const rescoreTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => { setPhase(result.phase_analysis); }, [result.task_id, result.phase_analysis]);
  useEffect(() => () => { if (rescoreTimer.current) clearTimeout(rescoreTimer.current); }, []);

  // Called by the comparison player when you move your contact point; debounced so dragging
  // the bar doesn't fire a request per pixel.
  const onUserContactChange = (sec: number) => {
    if (rescoreTimer.current) clearTimeout(rescoreTimer.current);
    rescoreTimer.current = setTimeout(async () => {
      setRescoring(true);
      try { setPhase(await rescorePhases(result.task_id, sec)); }
      catch { /* keep the previous phase scores */ }
      finally { setRescoring(false); }
    }, 500);
  };

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

      <div className="video-tabs">
        <button className={`vtab ${tab === "compare" ? "active" : ""}`} onClick={() => setTab("compare")}>
          {tx.compareTab}
        </button>
        <button className={`vtab ${tab === "overlay" ? "active" : ""}`} onClick={() => setTab("overlay")}>
          {tx.overlayTab}
        </button>
      </div>

      {tab === "compare" && (
        result.compare
          ? <ComparePlayer key={result.task_id} compare={result.compare} lang={lang}
              onUserContactChange={onUserContactChange} />
          : <div className="no-compare">{tx.noCompare}</div>
      )}
      {tab === "overlay" && annotated_video_url && (
        <VideoPlayer key={result.task_id} url={annotated_video_url} label={tx.skeletonVideo} />
      )}

      {phase && <PhasePanel phase={phase} lang={lang} rescoring={rescoring} />}
    </div>
  );
}

// Both clips play at the SAME selected speed (default 0.25x, the pro clip's speed).
// Timeline tau = seconds relative to contact (0 = ball strike) in media time. Whichever clip
// has more footage before contact (normally yours) starts first; the other waits on its first
// frame, so both strike the ball at the same instant.
// Your contact point can be moved by dragging your own progress bar (or ±frame buttons).
const SPEED_OPTIONS = [1, 0.5, 0.25, 0.1];
const DEFAULT_SPEED = 0.25;
const SPEED_KEY = "tv.compareSpeed";
const FRAME = 1 / 30;

function loadSpeed(): number {
  try {
    const v = parseFloat(localStorage.getItem(SPEED_KEY) ?? "");
    return SPEED_OPTIONS.includes(v) ? v : DEFAULT_SPEED;
  } catch { return DEFAULT_SPEED; }
}

function ComparePlayer({ compare, lang, onUserContactChange }: {
  compare: CompareInfo; lang: Lang; onUserContactChange?: (sec: number) => void;
}) {
  const tx = t[lang];
  const refRef  = useRef<HTMLVideoElement>(null);
  const userRef = useRef<HTMLVideoElement>(null);
  const [skeletonOnly, setSkeletonOnly] = useState(false);
  const [speed, setSpeedState] = useState(loadSpeed);
  const [durs, setDurs] = useState<{ ref: number; user: number }>({ ref: 0, user: 0 });
  const [playing, setPlaying] = useState(false);
  const [adjSec, setAdjSec] = useState(0);     // manual shift of YOUR contact point

  const refSrc  = skeletonOnly ? compare.ref_skeleton_url  : compare.ref_overlay_url;
  const userSrc = skeletonOnly ? compare.user_skeleton_url : compare.user_overlay_url;

  const rc = compare.ref_contact_sec;
  const uc = Math.min(Math.max(compare.user_contact_sec + adjSec, 0), durs.user || Infinity);
  const tauMin = -Math.max(rc, uc);
  const tauMax = Math.max((durs.ref || rc) - rc, (durs.user || uc) - uc, 0.001);
  const [tau, setTau] = useState(tauMin);

  const tauRef = useRef(tau);
  tauRef.current = tau;
  const gen = useRef(0);    // bumped on every seek/stop so a stale animation frame can't overwrite tau
  const clock = useRef({ tau0: 0, wall0: 0 });

  const setSpeed = (s: number) => {
    setSpeedState(s);
    try { localStorage.setItem(SPEED_KEY, String(s)); } catch { /* ignore */ }
  };

  const drive = (el: HTMLVideoElement | null, pos: number, dur: number, run: boolean) => {
    if (!el) return;
    const target = Math.min(Math.max(pos, 0), dur || Infinity);
    const inside = run && pos >= 0 && target < dur - 0.02;   // pos < 0: not started yet, hold first frame
    const rate = Math.max(speed, 0.0625);                    // browser playbackRate floor
    if (Math.abs(el.playbackRate - rate) > 1e-4) el.playbackRate = rate;
    if (inside) {
      // Only correct real drift; tiny lags from play() start-up are left alone to avoid seek thrash.
      const tol = speed < 0.0625 ? 0.02 : 0.06;
      if (!el.seeking && Math.abs(el.currentTime - target) > tol) el.currentTime = target;
      if (el.paused) el.play().catch(() => {});
    } else {
      if (!el.paused) el.pause();
      if (Math.abs(el.currentTime - target) > 0.01) el.currentTime = target;
    }
  };
  const driveBoth = (tl: number, run: boolean) => {
    drive(refRef.current,  rc + tl, durs.ref,  run);
    drive(userRef.current, uc + tl, durs.user, run);
  };

  // Playback loop (re-anchors when speed changes mid-play)
  useEffect(() => {
    if (!playing) return;
    const myGen = ++gen.current;
    const start = tauRef.current >= tauMax - 0.01 ? tauMin : tauRef.current;
    clock.current = { tau0: start, wall0: performance.now() };
    let raf = 0;
    const tick = () => {
      if (gen.current !== myGen) return;
      const tl = clock.current.tau0 + ((performance.now() - clock.current.wall0) / 1000) * speed;
      if (tl >= tauMax) {
        setTau(tauMax); setPlaying(false); driveBoth(tauMax, false);
        return;
      }
      setTau(tl);
      driveBoth(tl, true);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playing, speed, durs.ref, durs.user]);

  // When paused, keep both frames parked at the current position.
  useEffect(() => {
    if (!playing) driveBoth(tau, false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playing, tau, durs.ref, durs.user, adjSec]);

  // View switch: stop and go back to the start (a new analysis remounts this component).
  useEffect(() => {
    gen.current++; setPlaying(false); setTau(tauMin);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [compare, skeletonOnly]);

  const seek = (tl: number) => { gen.current++; setPlaying(false); setTau(Math.min(Math.max(tl, tauMin), tauMax)); };
  const toggle = () => { gen.current++; setPlaying(p => !p); };

  // Dragging YOUR progress bar moves only your video; the pro frame stays where it is, and the
  // new pairing becomes the alignment (i.e. it moves your contact point).
  const userNow = Math.min(Math.max(uc + tau, 0), durs.user || 0);
  const scrubUser = (v: number) => {
    gen.current++; setPlaying(false);
    setAdjSec(v - tau - compare.user_contact_sec);
  };
  const nudge = (frames: number) => {
    gen.current++; setPlaying(false);
    setAdjSec(a => a + frames * FRAME);
  };
  const resetAdj = () => { gen.current++; setPlaying(false); setAdjSec(0); setTau(0); };

  // Tell the parent when your contact point moves (not on first render).
  const firstUc = useRef(true);
  useEffect(() => {
    if (firstUc.current) { firstUc.current = false; return; }
    if (durs.user) onUserContactChange?.(uc);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [adjSec]);

  const fmt = (s: number) => (s >= 0 ? "+" : "") + s.toFixed(2) + "s";
  const pct = tauMax > tauMin ? ((0 - tauMin) / (tauMax - tauMin)) * 100 : 0;
  const adjFrames = Math.round(adjSec / FRAME);

  return (
    <div className="compare-section">
      <div className="compare-toolbar">
        <span className="sync-hint">⚡ {tx.syncHint} · {speed}x</span>
        <div className="compare-opts">
          <span className="opt-label">{tx.speedLabel}</span>
          {SPEED_OPTIONS.map((s) => (
            <button key={s} className={`speed-btn ${speed === s ? "active" : ""}`} onClick={() => setSpeed(s)}>
              {s}x
            </button>
          ))}
          <button
            className={`skeleton-toggle ${skeletonOnly ? "active" : ""}`}
            onClick={() => setSkeletonOnly(v => !v)}
          >
            {skeletonOnly ? "◉ Skeleton Only" : "◎ Skeleton Only"}
          </button>
        </div>
      </div>
      <div className="compare-grid">
        <div className="compare-pane">
          <div className="compare-label ref-label">{compare.ref_label}</div>
          <video ref={refRef} src={refSrc} className="compare-video" muted playsInline preload="auto"
            onLoadedMetadata={(e) => { const d = e.currentTarget.duration; setDurs(p => ({ ...p, ref: d })); }} />
        </div>
        <div className="compare-pane">
          <div className="compare-label user-label">{tx.yourServe}</div>
          <video ref={userRef} src={userSrc} className="compare-video" muted playsInline preload="auto"
            onLoadedMetadata={(e) => { const d = e.currentTarget.duration; setDurs(p => ({ ...p, user: d })); }} />
          <div className="user-scrub">
            <input type="range" className="scrub" min={0} max={durs.user || 1} step={FRAME / 2} value={userNow}
              onChange={(e) => scrubUser(parseFloat(e.target.value))} disabled={!durs.user}
              title={tx.userScrubHint} />
            <span className="time">{userNow.toFixed(2)}s</span>
          </div>
          <div className="nudge-row">
            <button className="speed-btn" onClick={() => nudge(-1)}>−1</button>
            <button className="speed-btn" onClick={() => nudge(1)}>+1</button>
            <span className="time">{adjFrames > 0 ? "+" : ""}{adjFrames} {tx.frames}</span>
            <button className="speed-btn" onClick={resetAdj} disabled={adjFrames === 0}>{tx.reset}</button>
          </div>
        </div>
      </div>
      <p className="scrub-hint">{tx.userScrubHint}</p>
      <div className="transport">
        <button className="speed-btn" onClick={toggle} disabled={!durs.ref || !durs.user}>
          {playing ? "❚❚" : "▶"}
        </button>
        <div className="scrub-wrap">
          <input type="range" className="scrub" min={tauMin} max={tauMax} step={0.005} value={tau}
            onChange={(e) => seek(parseFloat(e.target.value))} />
          <div className="contact-mark" style={{ left: `${pct}%` }} title={tx.contactPoint} />
        </div>
        <span className="time">{fmt(tau)}</span>
        <button className="speed-btn" onClick={() => seek(0)}>🎾 {tx.contactPoint}</button>
      </div>
    </div>
  );
}

function PhasePanel({ phase, lang, rescoring }: { phase: PhaseAnalysis; lang: Lang; rescoring: boolean }) {
  const tx = t[lang];
  const sign = (d: number) => (d > 0 ? "+" : "") + d.toFixed(0) + "°";
  return (
    <div className={`phase-section ${rescoring ? "busy" : ""}`}>
      <div className="phase-head">
        <div>
          <h3>{tx.phaseTitle}</h3>
          <p className="phase-sub">{tx.phaseSub} · {phase.ref_label}</p>
        </div>
        <div className="phase-overall">
          <span className="phase-overall-label">{rescoring ? tx.rescoring : tx.phaseOverall}</span>
          <span className={`phase-badge ${phase.overall_score >= 80 ? "good" : phase.overall_score >= 60 ? "warning" : "poor"}`}>
            {phase.overall_score}
          </span>
        </div>
      </div>
      <p className="scrub-hint">{tx.phaseHint}</p>

      {phase.phases.map((ph) => (
        <div key={ph.key} className="phase-card">
          <div className="phase-card-head">
            <span className="phase-name">{lang === "en" ? ph.name_en : ph.name_zh}</span>
            <span className={`phase-badge small ${ph.score >= 80 ? "good" : ph.score >= 60 ? "warning" : "poor"}`}>{ph.score}</span>
          </div>
          <div className="phase-shots">
            <figure>
              <img src={ph.ref_img_url} alt={`${phase.ref_label} ${ph.name_en}`} loading="lazy" />
              <figcaption className="ref-label">{phase.ref_label} · {ph.ref_time.toFixed(2)}s</figcaption>
            </figure>
            <figure>
              <img src={ph.user_img_url} alt={`${tx.yourServe} ${ph.name_en}`} loading="lazy" />
              <figcaption className="user-label">{tx.yourServe} · {ph.user_time.toFixed(2)}s</figcaption>
            </figure>
          </div>
          <table className="phase-table">
            <thead>
              <tr><th>{tx.colJoint}</th><th>{tx.colYou}</th><th>{phase.ref_label.split(" ").pop()}</th><th>{tx.colDiff}</th><th>{tx.colScore}</th></tr>
            </thead>
            <tbody>
              {ph.joints.map((j) => (
                <tr key={j.key} className={j.status}>
                  <td>{lang === "en" ? j.name_en : j.name_zh}</td>
                  <td>{j.user_angle.toFixed(0)}°</td>
                  <td>{j.ref_angle.toFixed(0)}°</td>
                  <td>{sign(j.diff)}</td>
                  <td className="jscore">{j.score}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ))}
    </div>
  );
}

const SPEEDS = [1, 0.5, 0.25, 0.1];

function VideoPlayer({ url, label }: { url: string; label: string }) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [speed, setSpeed] = useState(DEFAULT_SPEED);

  const setPlaybackRate = (s: number) => {
    setSpeed(s);
    if (videoRef.current) videoRef.current.playbackRate = s;
  };

  return (
    <div className="video-section">
      <div className="video-header">
        <h3>{label}</h3>
        <div className="speed-controls">
          {SPEEDS.map((s) => (
            <button
              key={s}
              className={`speed-btn ${speed === s ? "active" : ""}`}
              onClick={() => setPlaybackRate(s)}
            >
              {s}x
            </button>
          ))}
        </div>
      </div>
      <video
        ref={videoRef}
        controls
        src={url}
        className="annotated-video"
        onPlay={() => { if (videoRef.current) videoRef.current.playbackRate = speed; }}
      />
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
