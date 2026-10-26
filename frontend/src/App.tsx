import React, { useEffect, useState, useRef, useCallback } from "react";
import {
  AlertCircle,
  AlertTriangle,
  BarChart2,
  CheckCircle2,
  FastForward,
  Play,
  ShieldAlert,
  ShieldCheck,
  Square,
  Zap,
} from "lucide-react";
import type { HealthResponse, ReplayEvent, ScenarioItem } from "./events";
import { parseSSEStream } from "./sse";
import { ReplayTimeline, type SpeechSegment } from "./ReplayTimeline";
import { ResultsDashboard } from "./ResultsDashboard";
import { formatDelayLine } from "./resultLine";
import { formatEngineSummary, summarizeDemoSet, type EngineOutcome, type ScenarioOutcome } from "./demoTally";

interface DecisionEntry {
  checkpointMs: number;
  action: string;
  pDone: number;
  choice: string;
  latencyMs: number;
  arrivalMs: number;
}

interface SpeakerProfile {
  user_id: string;
  resumed_median_ms: number | null;
  finished_median_ms: number | null;
  filler_rate: number | null;
  has_history: boolean;
  rendered: string;
  regrets_count: number;
}

interface ReplayErrorInfo {
  message: string;
  engine?: string;
  checkpoint_ms?: number;
  state_key?: string;
  state_text?: string;
}

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<"demo" | "results">("demo");
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [scenarios, setScenarios] = useState<ScenarioItem[]>([]);
  const [selectedScenarioId, setSelectedScenarioId] = useState<string>("");
  const [speed, setSpeed] = useState<number>(1.0);

  // Replay Execution State
  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [isDemoSetRunning, setIsDemoSetRunning] = useState<boolean>(false);
  const [demoSetIndex, setDemoSetIndex] = useState<number>(0);
  const [demoOutcomes, setDemoOutcomes] = useState<ScenarioOutcome[]>([]);
  const [demoSetSize, setDemoSetSize] = useState<number>(0);
  // Outcome of the replay currently streaming, filled from its own events only.
  const runOutcomeRef = useRef<{
    fixed: EngineOutcome | null;
    patientListener: EngineOutcome | null;
    fixedRespondedAt: number | null;
    plRespondedAt: number | null;
    error: string | null;
  }>({ fixed: null, patientListener: null, fixedRespondedAt: null, plRespondedAt: null, error: null });

  const [replayError, setReplayError] = useState<ReplayErrorInfo | null>(null);

  // Live Engine Tracking
  const [fixedTranscript, setFixedTranscript] = useState<string>("");
  const [plTranscript, setPlTranscript] = useState<string>("");
  const fixedRespondedRef = useRef<boolean>(false);
  const plRespondedRef = useRef<boolean>(false);
  const [currentTimeMs, setCurrentTimeMs] = useState<number>(0);
  const [currentSilenceMs, setCurrentSilenceMs] = useState<number>(0);
  const [speechSegments, setSpeechSegments] = useState<SpeechSegment[]>([]);
  const [decisionsFeed, setDecisionsFeed] = useState<DecisionEntry[]>([]);

  const [fixedState, setFixedState] = useState<{
    responded: boolean;
    respondedAtMs: number | null;
    interrupted: boolean | null;
    responseDelayMs: number | null;
  }>({
    responded: false,
    respondedAtMs: null,
    interrupted: null,
    responseDelayMs: null,
  });

  const [plState, setPlState] = useState<{
    responded: boolean;
    respondedAtMs: number | null;
    interrupted: boolean | null;
    responseDelayMs: number | null;
  }>({
    responded: false,
    respondedAtMs: null,
    interrupted: null,
    responseDelayMs: null,
  });

  // Reply generated after Patient Listener decides RESPOND (Bedrock, or labeled mock)
  const [plReply, setPlReply] = useState<{
    text: string;
    source: "bedrock" | "mock" | null;
    modelId: string | null;
    firstTokenMs: number | null;
    done: boolean;
    error: string | null;
  }>({ text: "", source: null, modelId: null, firstTokenMs: null, done: false, error: null });

  const abortControllerRef = useRef<AbortController | null>(null);
  const fixedCutSilenceRef = useRef<number | null>(null);
  const activeSegmentStartRef = useRef<number | null>(null);

  const apiBase = (import.meta.env.VITE_API_URL as string) || "http://localhost:8000";

  // Fetch Health & Provenance
  const fetchHealth = useCallback(async () => {
    try {
      const resp = await fetch(`${apiBase}/health`);
      if (resp.ok) {
        const json = await resp.json();
        setHealth(json);
      }
    } catch (err) {
      console.error("Failed to fetch /health:", err);
    }
  }, [apiBase]);

  // Fetch Scenarios Catalog
  const fetchScenarios = useCallback(async () => {
    try {
      const resp = await fetch(`${apiBase}/scenarios`);
      if (resp.ok) {
        const json = await resp.json();
        const list: ScenarioItem[] = Array.isArray(json) ? json : json.scenarios || [];
        setScenarios(list);
        if (list.length > 0) {
          setSelectedScenarioId((curr) => {
            if (!curr) {
              const firstDemo = list.find((s) => s.demo) || list[0];
              return firstDemo.id;
            }
            return curr;
          });
        }
      }
    } catch (err) {
      console.error("Failed to fetch /scenarios:", err);
    }
  }, [apiBase]);

  useEffect(() => {
    fetchHealth();
    fetchScenarios();
  }, [fetchHealth, fetchScenarios]);

  const selectedScenario = scenarios.find((s) => s.id === selectedScenarioId);

  // Speaker pace profile (from prior sessions + logged regrets)
  const [profile, setProfile] = useState<SpeakerProfile | null>(null);
  const [regretNote, setRegretNote] = useState<string>("");

  const fetchProfile = async (user: string) => {
    try {
      const resp = await fetch(`${apiBase}/profile/${encodeURIComponent(user)}`);
      if (resp.ok) setProfile(await resp.json());
    } catch (err) {
      console.error("Failed to fetch /profile:", err);
    }
  };

  useEffect(() => {
    setRegretNote("");
    if (selectedScenario?.user) fetchProfile(selectedScenario.user);
    else setProfile(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedScenario?.user, apiBase]);

  // "The assistant cut me off here": log a regret for the fixed engine's cut-in
  const logRegret = async () => {
    if (!selectedScenario) return;
    const silence = fixedCutSilenceRef.current ?? 800;
    try {
      const resp = await fetch(`${apiBase}/regret`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_id: selectedScenario.user,
          session_id: selectedScenario.id,
          state_text: fixedTranscript,
          probs: {},
          silence_ms: silence,
          action: "RESPOND",
        }),
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      setRegretNote(`Logged: cut off after ${silence} ms of silence. Profile updated.`);
      await fetchProfile(selectedScenario.user);
    } catch (err) {
      setRegretNote(`Could not log regret (${err instanceof Error ? err.message : String(err)})`);
    }
  };
  const isMock = health?.cache_provenance?.is_mock ?? (health?.cache_provenance?.backend === "mock");

  const resetReplayState = () => {
    fixedRespondedRef.current = false;
    plRespondedRef.current = false;
    setFixedTranscript("");
    setPlTranscript("");
    setCurrentTimeMs(0);
    setCurrentSilenceMs(0);
    setSpeechSegments([]);
    setDecisionsFeed([]);
    setReplayError(null);
    activeSegmentStartRef.current = null;
    runOutcomeRef.current = { fixed: null, patientListener: null, fixedRespondedAt: null, plRespondedAt: null, error: null };
    setPlReply({ text: "", source: null, modelId: null, firstTokenMs: null, done: false, error: null });
    setFixedState({
      responded: false,
      respondedAtMs: null,
      interrupted: null,
      responseDelayMs: null,
    });
    setPlState({
      responded: false,
      respondedAtMs: null,
      interrupted: null,
      responseDelayMs: null,
    });
  };

  const handleEvent = (ev: ReplayEvent) => {
    if (ev.event === "error") {
      const shortMsg =
        ev.error === "cache_miss"
          ? `Cache miss: ${ev.engine ?? "patient_listener"} at ${ev.checkpoint_ms ?? "?"}ms`
          : ev.message || ev.error || "Cache miss error during replay";
      setReplayError({
        message: shortMsg,
        engine: ev.engine,
        checkpoint_ms: ev.checkpoint_ms,
        state_key: ev.state_key,
        state_text: ev.state_text,
      });
      runOutcomeRef.current.error = shortMsg;
      setIsRunning(false);
      return;
    }

    if (typeof ev.t_ms === "number") {
      setCurrentTimeMs(ev.t_ms);
    }

    switch (ev.event) {
      case "speechstarted": {
        activeSegmentStartRef.current = ev.t_ms;
        setCurrentSilenceMs(0);
        break;
      }
      case "speechdelta": {
        const text = ev.text;
        if (text) {
          if (!fixedRespondedRef.current) {
            setFixedTranscript((prev) => (prev ? `${prev} ${text}` : text));
          }
          if (!plRespondedRef.current) {
            setPlTranscript((prev) => (prev ? `${prev} ${text}` : text));
          }
        }
        break;
      }
      case "speechended": {
        if (activeSegmentStartRef.current !== null) {
          const seg: SpeechSegment = {
            startMs: activeSegmentStartRef.current,
            endMs: ev.t_ms,
          };
          setSpeechSegments((prev) => [...prev, seg]);
          activeSegmentStartRef.current = null;
        }
        break;
      }
      case "decision_completed": {
        setCurrentSilenceMs(ev.checkpoint_ms);
        setDecisionsFeed((prev) => [
          ...prev,
          {
            checkpointMs: ev.checkpoint_ms,
            action: ev.action,
            pDone: ev.p_done,
            choice: ev.choice,
            latencyMs: ev.latency_ms,
            arrivalMs: ev.t_ms,
          },
        ]);
        break;
      }
      case "decision_dropped": {
        setCurrentSilenceMs(0);
        break;
      }
      case "respond": {
        if (ev.engine === "fixed") {
          fixedRespondedRef.current = true;
          runOutcomeRef.current.fixedRespondedAt = ev.t_ms;
          fixedCutSilenceRef.current = (ev as { silence_ms?: number }).silence_ms ?? null;
          setFixedState((prev) => ({
            ...prev,
            responded: true,
            respondedAtMs: ev.t_ms,
          }));
        } else if (ev.engine === "patient_listener") {
          plRespondedRef.current = true;
          runOutcomeRef.current.plRespondedAt = ev.t_ms;
          setPlState((prev) => ({
            ...prev,
            responded: true,
            respondedAtMs: ev.t_ms,
          }));
        }
        break;
      }
      case "reply_chunk": {
        setPlReply((prev) => ({
          ...prev,
          text: prev.text + ev.text,
          source: ev.reply_source,
          modelId: ev.model_id,
        }));
        break;
      }
      case "reply_done": {
        setPlReply({
          text: ev.full_reply,
          source: ev.reply_source,
          modelId: ev.model_id,
          firstTokenMs: ev.first_token_ms,
          done: true,
          error: ev.error,
        });
        break;
      }
      case "final": {
        if (ev.engine === "fixed") {
          if (ev.interrupted) {
            fixedRespondedRef.current = true;
          }
          setFixedState((prev) => ({
            ...prev,
            interrupted: ev.interrupted,
            responseDelayMs: ev.response_delay_ms,
          }));
          runOutcomeRef.current.fixed = {
            interrupted: ev.interrupted,
            respondedAtMs: runOutcomeRef.current.fixedRespondedAt,
            responseDelayMs: ev.response_delay_ms ?? null,
          };
        } else if (ev.engine === "patient_listener") {
          if (ev.interrupted) {
            plRespondedRef.current = true;
          }
          setPlState((prev) => ({
            ...prev,
            interrupted: ev.interrupted,
            responseDelayMs: ev.response_delay_ms,
          }));
          runOutcomeRef.current.patientListener = {
            interrupted: ev.interrupted,
            respondedAtMs: runOutcomeRef.current.plRespondedAt,
            responseDelayMs: ev.response_delay_ms ?? null,
          };
        }
        break;
      }
    }
  };

  const runReplay = async (scenarioId: string, replaySpeed: number): Promise<boolean> => {
    resetReplayState();
    setIsRunning(true);

    const controller = new AbortController();
    abortControllerRef.current = controller;

    try {
      const response = await fetch(`${apiBase}/replay/${scenarioId}?speed=${replaySpeed}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: controller.signal,
      });

      if (!response.ok) {
        throw new Error(`Replay request failed with HTTP ${response.status}: ${response.statusText}`);
      }

      if (!response.body) {
        throw new Error("No ReadableStream in replay response");
      }

      await parseSSEStream(response.body, handleEvent, controller.signal);
      setIsRunning(false);
      return true;
    } catch (err: unknown) {
      if ((err as Error)?.name === "AbortError") {
        console.log("Replay aborted by user");
      } else {
        const msg = err instanceof Error ? err.message : String(err);
        setReplayError({ message: msg });
      }
      setIsRunning(false);
      return false;
    }
  };

  const startReplay = () => {
    if (!selectedScenarioId) return;
    runReplay(selectedScenarioId, speed);
  };

  const cancelReplay = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    setIsRunning(false);
    setIsDemoSetRunning(false);
  };

  // Play Demo Set Sequential Runner
  const playDemoSet = async () => {
    const demoScenarios = scenarios.filter((s) => s.demo);
    if (demoScenarios.length === 0) return;

    setIsDemoSetRunning(true);
    setDemoOutcomes([]);
    setDemoSetSize(demoScenarios.length);

    for (let i = 0; i < demoScenarios.length; i++) {
      setDemoSetIndex(i);
      const sc = demoScenarios[i];
      setSelectedScenarioId(sc.id);

      // false only on cancel or a failed request; a stream error event still returns true
      const success = await runReplay(sc.id, speed);
      if (!success) break;
      const r = runOutcomeRef.current;
      const outcome: ScenarioOutcome = {
        scenarioId: sc.id,
        trueEndMs: sc.true_end_t_ms,
        fixed: r.fixed,
        patientListener: r.patientListener,
        error: r.error,
      };
      setDemoOutcomes((prev) => [...prev, outcome]);

      // Small pause between scenarios
      await new Promise((r) => setTimeout(r, 600));
    }

    setIsDemoSetRunning(false);
  };

  return (
    <div className="min-h-screen text-slate-100 flex flex-col font-sans">
      {/* Main App Header */}
      <header className="border-b border-slate-800 bg-slate-950/80 backdrop-blur sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between gap-6">
          <div className="flex items-center gap-8">
            <div className="flex items-center gap-2.5">
              <div className="w-8 h-8 rounded-md bg-indigo-300 flex items-center justify-center">
                <Zap className="w-4 h-4 text-slate-950" />
              </div>
              <span className="font-mono text-[13px] font-semibold uppercase tracking-[0.12em] text-slate-50 leading-tight">
                Patient
                <br />
                Listener_
              </span>
            </div>

            {/* Tab Navigation: monospace uppercase links */}
            <nav className="flex items-center gap-7">
              {([
                ["demo", "Side-by-side", Play],
                ["results", "Results", BarChart2],
              ] as const).map(([tab, label, Icon]) => (
                <button
                  key={tab}
                  onClick={() => setActiveTab(tab)}
                  className={`flex items-center gap-2 py-5 font-mono text-[13px] uppercase tracking-[0.14em] border-b-2 transition ${
                    activeTab === tab
                      ? "text-slate-50 border-indigo-300"
                      : "text-slate-400 border-transparent hover:text-slate-200"
                  }`}
                >
                  <Icon className="w-3.5 h-3.5" />
                  {label}
                </button>
              ))}
            </nav>
          </div>

          <span className="hidden md:inline font-mono text-[11px] uppercase tracking-[0.14em] text-slate-400">
            #strands-decider-2b · #policy
          </span>
        </div>
      </header>

      {/* Hero: headline with live turn switch + provenance card */}
      <section className="max-w-7xl w-full mx-auto px-6 pt-10 pb-6 grid lg:grid-cols-[1fr_380px] gap-8 items-start">
        <div>
          <div className="inline-flex items-center gap-2 border border-slate-700 rounded-md px-3 py-1.5 font-mono text-[11px] uppercase tracking-[0.14em] text-slate-400">
            <span className={`w-1.5 h-1.5 rounded-full ${isRunning ? "bg-indigo-300 animate-pulse" : "bg-slate-500"}`} />
            {isRunning ? "Replay running" : "Voice turn-taking · Strands Decider 2B"}
          </div>
          <h1 className="mt-5 font-display text-5xl md:text-6xl font-semibold leading-[1.02] text-slate-50">
            Patient Listener{" "}
            <span
              aria-label={plState.responded ? "Responding" : "Listening"}
              className={`inline-flex align-middle w-[86px] h-[46px] rounded-full p-[5px] transition-colors duration-300 ${
                plState.responded ? "bg-indigo-400" : "bg-slate-800"
              }`}
            >
              <span
                className={`w-9 h-9 rounded-full transition-transform duration-300 ${
                  plState.responded ? "translate-x-[40px] bg-slate-50" : "translate-x-0 bg-slate-500"
                }`}
              />
            </span>
          </h1>
          <p className="mt-5 max-w-2xl text-lg leading-relaxed text-slate-200">
            Fixed silence timeouts cut people off mid-thought. Patient Listener asks a small decision model
            whether you're actually done, and waits when you're not.
          </p>
        </div>

        {/* Provenance card */}
        <div className="border border-slate-800 bg-slate-950/70 rounded-md">
          <div className="flex items-center justify-between px-5 py-3 border-b border-slate-800 font-mono text-[11px] uppercase tracking-[0.14em] text-slate-400">
            <span>Decision source</span>
            {isMock ? (
              <span className="flex items-center gap-1.5 text-rose-400">
                <ShieldAlert className="w-3.5 h-3.5" /> Mock
              </span>
            ) : (
              <span className="flex items-center gap-1.5 text-emerald-400">
                <ShieldCheck className="w-3.5 h-3.5" /> Real
              </span>
            )}
          </div>
          <div className="px-5 py-4 space-y-3">
            {isMock ? (
              <p className="text-sm text-rose-300">
                MOCK-DECIDER, not valid for the demo. The real GPU decision cache is not installed.
              </p>
            ) : (
              <>
                <div className="text-xl font-semibold text-slate-50">Strands Decider 2B</div>
                <div className="font-mono text-xs text-slate-400 break-all">
                  {health?.cache_provenance?.model_id ?? "cached decisions"}
                </div>
                <div className="grid grid-cols-2 gap-3 pt-2 border-t border-slate-800">
                  <div>
                    <div className="label-mono">Latency</div>
                    <div className="text-slate-100 text-sm mt-0.5">
                      {health?.cache_provenance?.decision_latency_ms
                        ? `${Math.round(health.cache_provenance.decision_latency_ms)} ms`
                        : "n/a"}
                    </div>
                  </div>
                  <div>
                    <div className="label-mono">Hardware</div>
                    <div className="text-slate-100 text-sm mt-0.5">{health?.cache_provenance?.hardware ?? "n/a"}</div>
                  </div>
                </div>
              </>
            )}
          </div>
        </div>
      </section>

      {/* Main Body Content */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-6 pb-10">
        {activeTab === "results" ? (
          <ResultsDashboard apiBase={apiBase} />
        ) : (
          <div className="space-y-6">
            {/* Control Strip (Scenario Picker + Speed + Run Controls) */}
            <div className="bg-slate-950/70 border border-slate-800 rounded-md p-5 flex flex-wrap items-end justify-between gap-5">
              {/* Scenario Picker */}
              <div className="flex-1 min-w-[320px]">
                <label className="block label-mono mb-2">
                  Select Conversational Scenario
                </label>
                <select
                  disabled={isRunning}
                  value={selectedScenarioId}
                  onChange={(e) => {
                    setSelectedScenarioId(e.target.value);
                    resetReplayState();
                  }}
                  className="w-full bg-slate-900 border border-slate-700 text-slate-100 text-sm rounded-md px-3 py-2.5 focus:outline-none focus:border-indigo-300 disabled:opacity-50"
                >
                  {scenarios.map((sc) => (
                    <option key={sc.id} value={sc.id}>
                      {sc.demo ? "★ [DEMO] " : "  "}
                      {sc.persona} ({sc.user}): &ldquo;{sc.text_preview.slice(0, 75)}
                      {sc.text_preview.length > 75 ? "..." : ""}&rdquo;
                    </option>
                  ))}
                </select>
              </div>

              {/* Speed Selector */}
              <div>
                <label className="block label-mono mb-2">
                  Replay Speed
                </label>
                <div className="flex items-center gap-1 bg-slate-950 p-1 rounded-lg border border-slate-800">
                  {[1.0, 2.0, 4.0].map((s) => (
                    <button
                      key={s}
                      disabled={isRunning}
                      onClick={() => setSpeed(s)}
                      className={`px-3 py-1.5 font-mono text-xs rounded ${
                        speed === s
                          ? "bg-indigo-600 text-white"
                          : "text-slate-400 hover:text-slate-200"
                      } disabled:opacity-50`}
                    >
                      {s}x
                    </button>
                  ))}
                </div>
              </div>

              {/* Action Buttons */}
              <div className="flex items-center gap-2">
                {!isRunning ? (
                  <button
                    onClick={startReplay}
                    disabled={!selectedScenarioId}
                    className="btn-accent flex items-center gap-2 px-5 py-2.5 rounded-md text-sm"
                  >
                    <Play className="w-4 h-4 fill-current" />
                    Start Replay
                  </button>
                ) : (
                  <button
                    onClick={cancelReplay}
                    className="flex items-center gap-2 px-5 py-2.5 rounded-md text-sm font-medium bg-rose-400 text-slate-950 hover:bg-rose-300 transition"
                  >
                    <Square className="w-4 h-4 fill-current" />
                    Cancel Replay
                  </button>
                )}

                <button
                  onClick={playDemoSet}
                  disabled={isRunning}
                  className="btn-outline flex items-center gap-2 px-4 py-2.5 rounded-md text-sm"
                  title="Run all demo-flagged scenarios in sequence"
                >
                  <FastForward className="w-4 h-4 text-indigo-400" />
                  Play Demo Set
                </button>
              </div>
            </div>

            {/* Running Demo Set Tally Banner */}
            {(isDemoSetRunning || demoOutcomes.length > 0) && (() => {
              const tally = summarizeDemoSet(demoOutcomes);
              return (
                <div className="bg-indigo-950/60 border border-indigo-700/60 rounded-xl p-3 flex flex-col gap-2 text-sm text-indigo-200 shadow-md">
                  <div className="flex items-center gap-2">
                    <FastForward className={`w-4 h-4 text-indigo-400 ${isDemoSetRunning ? "animate-pulse" : ""}`} />
                    <span className="font-semibold">
                      {isDemoSetRunning
                        ? `Demo set running: scenario ${demoSetIndex + 1} of ${demoSetSize}`
                        : `Demo set finished: ${tally.evaluated} of ${demoSetSize} scenarios evaluated`}
                      {tally.errored > 0 ? ` (${tally.errored} errored)` : ""}
                    </span>
                  </div>
                  <div className="font-mono text-xs flex flex-col gap-1">
                    <span>
                      Fixed 800 ms: <strong className="text-rose-400">{tally.fixed.cutIns} cut-ins</strong>;{" "}
                      {formatEngineSummary(tally.fixed)}
                    </span>
                    <span>
                      Patient Listener: <strong className="text-emerald-400">{tally.patientListener.cutIns} cut-ins</strong>;{" "}
                      {formatEngineSummary(tally.patientListener)}
                    </span>
                  </div>
                </div>
              );
            })()}

            {/* Error Banner */}
            {replayError && (
              <div className="p-4 bg-rose-950/80 border border-rose-500 rounded-xl text-rose-200 text-xs shadow-lg">
                <div className="flex items-start justify-between gap-4">
                  <div className="flex items-center gap-2">
                    <AlertCircle className="w-4 h-4 text-rose-400 shrink-0" />
                    <span className="font-semibold text-rose-100">{replayError.message}</span>
                  </div>
                  <button
                    onClick={() => setReplayError(null)}
                    className="text-xs text-rose-400 hover:text-rose-200 underline shrink-0"
                  >
                    Dismiss
                  </button>
                </div>

                {(replayError.state_key || replayError.state_text) && (
                  <details className="mt-2.5 text-[11px] text-rose-300/80">
                    <summary className="cursor-pointer font-medium hover:text-rose-100 select-none">
                      Show details (state key & text)
                    </summary>
                    <div className="mt-2 p-2.5 bg-slate-950/90 border border-rose-900/60 rounded font-mono text-[11px] space-y-1.5 break-all text-slate-300">
                      {replayError.state_key && (
                        <div>
                          <span className="text-slate-400 font-semibold">State Key: </span>
                          <span className="text-amber-300">{replayError.state_key}</span>
                        </div>
                      )}
                      {replayError.state_text && (
                        <div>
                          <span className="text-slate-400 font-semibold">State Text: </span>
                          <span className="text-slate-200 font-sans">"{replayError.state_text}"</span>
                        </div>
                      )}
                    </div>
                  </details>
                )}
              </div>
            )}

            {/* Selected Scenario Metadata Card */}
            {selectedScenario && (
              <div className="bg-slate-900/40 border border-slate-800/80 rounded-xl px-4 py-2.5 flex flex-wrap items-center justify-between text-xs text-slate-300">
                <div className="flex items-center gap-4">
                  <span>
                    <strong className="text-slate-400 font-normal">Persona:</strong>{" "}
                    <span className="font-semibold text-indigo-300">{selectedScenario.persona}</span>
                  </span>
                  <span>
                    <strong className="text-slate-400 font-normal">Speaker:</strong>{" "}
                    <span className="font-mono text-slate-200">{selectedScenario.user}</span>
                  </span>
                  <span>
                    <strong className="text-slate-400 font-normal">Category:</strong>{" "}
                    <span className="capitalize text-slate-200">{selectedScenario.category}</span>
                  </span>
                </div>
                <div className="flex items-center gap-3 text-slate-400 text-[11px]">
                  <span>
                    True speech end: <strong className="text-amber-300">{Math.round(selectedScenario.true_end_t_ms)} ms</strong>
                  </span>
                  <span>
                    Pause silence: <strong className="text-slate-200">{Math.round(selectedScenario.silence_ms)} ms</strong>
                  </span>
                </div>
              </div>
            )}

            {/* Speaker pace profile: learned from this user's earlier sessions and logged regrets */}
            {profile && (
              <div className="bg-slate-950/70 border border-slate-800 rounded-md px-5 py-3 flex flex-wrap items-center gap-x-8 gap-y-2 text-sm">
                <span className="label-mono">Speaker profile</span>
                {profile.has_history ? (
                  <>
                    <span className="text-slate-300">
                      Pauses then keeps talking: <strong className="text-slate-50">
                        {profile.resumed_median_ms !== null ? `${Math.round(profile.resumed_median_ms)} ms` : "n/a"}
                      </strong>{" "}
                      <span className="text-slate-500">median</span>
                    </span>
                    <span className="text-slate-300">
                      Pause when finished: <strong className="text-slate-50">
                        {profile.finished_median_ms !== null ? `${Math.round(profile.finished_median_ms)} ms` : "n/a"}
                      </strong>
                    </span>
                    <span className="text-slate-300">
                      Filler rate: <strong className="text-slate-50">
                        {profile.filler_rate !== null ? `${Math.round(profile.filler_rate * 100)}%` : "n/a"}
                      </strong>
                    </span>
                  </>
                ) : (
                  <span className="text-slate-400">No earlier sessions for this speaker (cold start).</span>
                )}
                <span className="text-slate-300">
                  Regrets logged: <strong className="text-slate-50">{profile.regrets_count}</strong>
                </span>
              </div>
            )}

            {/* Side-by-Side Dual Engine Comparison Panels */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Left Panel: Fixed 800ms Timeout */}
              <div
                className={`bg-slate-900/90 border rounded-xl p-5 shadow-xl flex flex-col justify-between transition-all ${
                  fixedState.interrupted
                    ? "border-rose-500/80 ring-1 ring-rose-500/40"
                    : "border-slate-700/80"
                }`}
              >
                <div>
                  <div className="flex items-center justify-between border-b border-slate-800 pb-3 mb-4">
                    <div>
                      <h2 className="text-base font-bold text-white flex items-center gap-2">
                        Fixed 800 ms Timeout
                        {fixedState.interrupted && (
                          <span className="px-2 py-0.5 rounded bg-rose-600 text-white font-bold text-[10px] tracking-wider uppercase animate-pulse">
                            INTERRUPTED
                          </span>
                        )}
                      </h2>
                      <p className="text-[11px] text-slate-400 mt-0.5">
                        Static timeout baseline: cuts in after 800 ms of silence regardless of sentence completion
                      </p>
                    </div>
                  </div>

                  {/* Transcript Display Box */}
                  <div className="mb-4">
                    <span className="text-[10px] uppercase font-bold text-slate-500 tracking-wider block mb-1">
                      Live Spoken Audio Transcript
                    </span>
                    <div className="bg-slate-950 rounded-lg p-3 min-h-[90px] border border-slate-800/80 text-sm leading-relaxed font-sans text-slate-200">
                      {fixedTranscript || (
                        <span className="text-slate-600 italic">Waiting for speech input...</span>
                      )}
                    </div>
                  </div>

                  {/* Live Silence Bar */}
                  <div className="mb-4">
                    <div className="flex items-center justify-between text-[11px] text-slate-400 mb-1">
                      <span>Live Pause Silence Duration</span>
                      <span className="font-mono text-slate-200">{Math.round(currentSilenceMs)} ms / 800 ms threshold</span>
                    </div>
                    <div className="w-full h-2.5 bg-slate-950 rounded-full overflow-hidden border border-slate-800 relative">
                      <div
                        className={`h-full transition-all duration-100 ${
                          currentSilenceMs >= 800 ? "bg-rose-500" : "bg-amber-400"
                        }`}
                        style={{ width: `${Math.min(100, (currentSilenceMs / 800) * 100)}%` }}
                      />
                    </div>
                  </div>

                  {/* Engine Action Marker */}
                  <div className="p-3 bg-slate-950/60 rounded-lg border border-slate-800/60 text-xs">
                    {fixedState.responded ? (
                      <div className="flex items-center gap-2 text-rose-300 font-medium">
                        <AlertTriangle className="w-4 h-4 text-rose-400" />
                        <span>
                          Assistant started replying at <strong>{Math.round(fixedState.respondedAtMs ?? 0)} ms</strong>
                        </span>
                      </div>
                    ) : (
                      <span className="text-slate-500 text-xs italic">
                        {isRunning ? "Listening for speech pause..." : "Idle"}
                      </span>
                    )}
                  </div>
                </div>

                {/* Left Panel Final Result Box */}
                <div className="mt-4 pt-3 border-t border-slate-800/80 flex items-center justify-between text-xs">
                  <div>
                    <span className="text-slate-400">Interrupted mid-thought: </span>
                    {fixedState.interrupted === null ? (
                      <span className="text-slate-500">—</span>
                    ) : fixedState.interrupted ? (
                      <strong className="text-rose-400">YES (Cut user off)</strong>
                    ) : (
                      <strong className="text-emerald-400">NO</strong>
                    )}
                  </div>
                  <div>
                    <span className="text-slate-400">Response timing: </span>
                    <strong className="text-slate-200">
                      {formatDelayLine(fixedState.respondedAtMs, selectedScenario?.true_end_t_ms)}
                    </strong>
                  </div>
                </div>

                {/* Regret feedback: feeds the speaker's pace profile */}
                {fixedState.interrupted && !isRunning && (
                  <div className="mt-3 flex flex-wrap items-center gap-3">
                    <button onClick={logRegret} className="btn-outline px-3 py-1.5 rounded-md text-xs">
                      Log this cut-off as a regret
                    </button>
                    {regretNote && <span className="text-xs text-slate-300">{regretNote}</span>}
                  </div>
                )}
              </div>

              {/* Right Panel: Patient Listener (Decider 2B) */}
              <div
                className={`bg-slate-900/90 border rounded-xl p-5 shadow-xl flex flex-col justify-between transition-all ${
                  plState.responded && !plState.interrupted
                    ? "border-emerald-500/80 ring-1 ring-emerald-500/40"
                    : "border-slate-700/80"
                }`}
              >
                <div>
                  <div className="flex items-center justify-between border-b border-slate-800 pb-3 mb-4">
                    <div>
                      <h2 className="text-base font-bold text-white flex items-center gap-2">
                        Patient Listener (Decider 2B)
                        {plState.responded && !plState.interrupted && (
                          <span className="px-2 py-0.5 rounded bg-emerald-600 text-white font-bold text-[10px] tracking-wider uppercase">
                            PATIENT WAIT
                          </span>
                        )}
                        {plState.interrupted && (
                          <span className="px-2 py-0.5 rounded bg-rose-600 text-white font-bold text-[10px] tracking-wider uppercase">
                            INTERRUPTED
                          </span>
                        )}
                      </h2>
                      <p className="text-[11px] text-slate-400 mt-0.5">
                        Speculative System One decider: analyzes linguistic completion during pause checkpoints
                      </p>
                    </div>
                  </div>

                  {/* Transcript Display Box */}
                  <div className="mb-4">
                    <span className="text-[10px] uppercase font-bold text-slate-500 tracking-wider block mb-1">
                      Live Spoken Audio Transcript
                    </span>
                    <div className="bg-slate-950 rounded-lg p-3 min-h-[90px] border border-slate-800/80 text-sm leading-relaxed font-sans text-slate-200">
                      {plTranscript || (
                        <span className="text-slate-600 italic">Waiting for speech input...</span>
                      )}
                    </div>
                  </div>

                  {/* Live Silence Bar */}
                  <div className="mb-4">
                    <div className="flex items-center justify-between text-[11px] text-slate-400 mb-1">
                      <span>Silence Duration & Checkpoints</span>
                      <span className="font-mono text-slate-200">{Math.round(currentSilenceMs)} ms / 2200 ms max</span>
                    </div>
                    <div className="w-full h-2.5 bg-slate-950 rounded-full overflow-hidden border border-slate-800 relative">
                      <div
                        className="h-full bg-emerald-400 transition-all duration-100"
                        style={{ width: `${Math.min(100, (currentSilenceMs / 2200) * 100)}%` }}
                      />
                    </div>
                  </div>

                  {/* Decision Checkpoint Stream Feed */}
                  <div className="mb-3">
                    <span className="text-[10px] uppercase font-bold text-slate-500 tracking-wider block mb-1.5 flex items-center gap-1.5">
                      <Zap className="w-3.5 h-3.5 text-emerald-400" />
                      Speculative Checkpoint Decision Feed
                    </span>
                    <div className="bg-slate-950 rounded-lg p-2.5 max-h-[110px] overflow-y-auto border border-slate-800/80 space-y-1.5 font-mono text-[11px]">
                      {decisionsFeed.length === 0 ? (
                        <div className="text-slate-600 italic py-2 text-center font-sans">
                          No decision checkpoints triggered yet
                        </div>
                      ) : (
                        decisionsFeed.map((df, idx) => (
                          <div
                            key={idx}
                            className="flex items-center justify-between px-2 py-1 bg-slate-900/60 rounded border border-slate-800/60"
                          >
                            <span className="text-indigo-300">cp: {df.checkpointMs}ms</span>
                            <span
                              className={`font-bold px-1.5 py-0.2 rounded text-[10px] ${
                                df.action === "WAIT"
                                  ? "bg-amber-500/20 text-amber-300 border border-amber-500/30"
                                  : "bg-emerald-500/20 text-emerald-300 border border-emerald-500/30"
                              }`}
                            >
                              {df.action}
                            </span>
                            <span className="text-slate-400">p(done): {(df.pDone * 100).toFixed(0)}%</span>
                            <span className="text-slate-500 text-[10px]">lat: {df.latencyMs.toFixed(0)}ms</span>
                          </div>
                        ))
                      )}
                    </div>
                  </div>

                  {/* Engine Action Marker */}
                  <div className="p-3 bg-slate-950/60 rounded-lg border border-slate-800/60 text-xs">
                    {plState.responded ? (
                      <div className="flex items-center gap-2 text-emerald-300 font-medium">
                        <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                        <span>
                          Assistant started replying at <strong>{Math.round(plState.respondedAtMs ?? 0)} ms</strong>
                        </span>
                      </div>
                    ) : (
                      <span className="text-slate-500 text-xs italic">
                        {isRunning ? "Evaluating speculative pause checkpoints..." : "Idle"}
                      </span>
                    )}
                  </div>

                  {/* Assistant reply (Amazon Bedrock, or a labeled mock) */}
                  {(plReply.text || (plState.responded && !plState.interrupted)) && (
                    <div className="mt-3 rounded-md border border-indigo-300/30 bg-indigo-950/30 p-3">
                      <div className="label-mono mb-1.5 flex items-center justify-between gap-2">
                        <span>Assistant reply</span>
                        <span className={plReply.source === "bedrock" ? "text-indigo-300" : "text-amber-300"}>
                          {plReply.source === "bedrock"
                            ? `Amazon Bedrock · ${plReply.modelId?.replace(/^us\.anthropic\./, "") ?? ""}`
                            : plReply.source === "mock"
                              ? "Mock reply"
                              : "Generating…"}
                        </span>
                      </div>
                      <p className="text-sm leading-relaxed text-slate-100 min-h-[1.25rem]">
                        {plReply.text || <span className="text-slate-500 italic">waiting for first token…</span>}
                      </p>
                      {plReply.done && (
                        <div className="mt-1.5 font-mono text-[11px] text-slate-400">
                          {plReply.firstTokenMs !== null ? `first token ${Math.round(plReply.firstTokenMs)} ms` : ""}
                          {plReply.error ? ` · Bedrock failed, mock used (${plReply.error.slice(0, 60)})` : ""}
                        </div>
                      )}
                    </div>
                  )}
                </div>

                {/* Right Panel Final Result Box */}
                <div className="mt-4 pt-3 border-t border-slate-800/80 flex items-center justify-between text-xs">
                  <div>
                    <span className="text-slate-400">Interrupted mid-thought: </span>
                    {plState.interrupted === null ? (
                      <span className="text-slate-500">—</span>
                    ) : plState.interrupted ? (
                      <strong className="text-rose-400">YES</strong>
                    ) : (
                      <strong className="text-emerald-400">NO (Patiently waited)</strong>
                    )}
                  </div>
                  <div>
                    <span className="text-slate-400">Response timing: </span>
                    <strong className="text-slate-200">
                      {formatDelayLine(plState.respondedAtMs, selectedScenario?.true_end_t_ms)}
                    </strong>
                  </div>
                </div>
              </div>
            </div>

            {/* Shared Visual Timeline Strip */}
            <ReplayTimeline
              speechSegments={speechSegments}
              trueEndTimeMs={selectedScenario?.true_end_t_ms ?? 0}
              currentTimeMs={currentTimeMs}
              fixedResult={{
                respondedAtMs: fixedState.respondedAtMs,
                interrupted: fixedState.interrupted,
                responseDelayMs: fixedState.responseDelayMs,
              }}
              plResult={{
                respondedAtMs: plState.respondedAtMs,
                interrupted: plState.interrupted,
                responseDelayMs: plState.responseDelayMs,
              }}
              streamError={replayError?.message ?? null}
            />
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800/80 py-4 px-6 text-center text-xs text-slate-500">
        Patient Listener — System One Voice Conversational Turn-Taking Benchmark Demo
      </footer>
    </div>
  );
};

export default App;
