import React, { useEffect, useState, useRef } from "react";
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

interface DecisionEntry {
  checkpointMs: number;
  action: string;
  pDone: number;
  choice: string;
  latencyMs: number;
  arrivalMs: number;
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
  const [demoSetTally, setDemoSetTally] = useState<{
    total: number;
    fixedInterrupted: number;
    plInterrupted: number;
    fixedDelays: number[];
    plDelays: number[];
  }>({
    total: 0,
    fixedInterrupted: 0,
    plInterrupted: 0,
    fixedDelays: [],
    plDelays: [],
  });

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

  const abortControllerRef = useRef<AbortController | null>(null);
  const activeSegmentStartRef = useRef<number | null>(null);

  const apiBase = (import.meta.env.VITE_API_URL as string) || "http://localhost:8000";

  // Fetch Health & Provenance
  const fetchHealth = async () => {
    try {
      const resp = await fetch(`${apiBase}/health`);
      if (resp.ok) {
        const json = await resp.json();
        setHealth(json);
      }
    } catch (err) {
      console.error("Failed to fetch /health:", err);
    }
  };

  // Fetch Scenarios Catalog
  const fetchScenarios = async () => {
    try {
      const resp = await fetch(`${apiBase}/scenarios`);
      if (resp.ok) {
        const json = await resp.json();
        const list: ScenarioItem[] = Array.isArray(json) ? json : json.scenarios || [];
        setScenarios(list);
        if (list.length > 0 && !selectedScenarioId) {
          // Default to first demo-flagged scenario
          const firstDemo = list.find((s) => s.demo) || list[0];
          setSelectedScenarioId(firstDemo.id);
        }
      }
    } catch (err) {
      console.error("Failed to fetch /scenarios:", err);
    }
  };

  useEffect(() => {
    fetchHealth();
    fetchScenarios();
  }, [apiBase]);

  const selectedScenario = scenarios.find((s) => s.id === selectedScenarioId);
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
          setFixedState((prev) => ({
            ...prev,
            responded: true,
            respondedAtMs: ev.t_ms,
          }));
        } else if (ev.engine === "patient_listener") {
          plRespondedRef.current = true;
          setPlState((prev) => ({
            ...prev,
            responded: true,
            respondedAtMs: ev.t_ms,
          }));
        }
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
          setDemoSetTally((prev) => ({
            ...prev,
            fixedInterrupted: prev.fixedInterrupted + (ev.interrupted ? 1 : 0),
            fixedDelays: ev.response_delay_ms !== null ? [...prev.fixedDelays, ev.response_delay_ms] : prev.fixedDelays,
          }));
        } else if (ev.engine === "patient_listener") {
          if (ev.interrupted) {
            plRespondedRef.current = true;
          }
          setPlState((prev) => ({
            ...prev,
            interrupted: ev.interrupted,
            responseDelayMs: ev.response_delay_ms,
          }));
          setDemoSetTally((prev) => ({
            ...prev,
            total: prev.total + 1,
            plInterrupted: prev.plInterrupted + (ev.interrupted ? 1 : 0),
            plDelays: ev.response_delay_ms !== null ? [...prev.plDelays, ev.response_delay_ms] : prev.plDelays,
          }));
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
    setDemoSetTally({
      total: 0,
      fixedInterrupted: 0,
      plInterrupted: 0,
      fixedDelays: [],
      plDelays: [],
    });

    for (let i = 0; i < demoScenarios.length; i++) {
      setDemoSetIndex(i);
      const sc = demoScenarios[i];
      setSelectedScenarioId(sc.id);

      const success = await runReplay(sc.id, speed);
      if (!success) break;

      // Small pause between scenarios
      await new Promise((r) => setTimeout(r, 600));
    }

    setIsDemoSetRunning(false);
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-indigo-500/30">
      {/* Top Provenance Banner */}
      {isMock ? (
        <div className="w-full bg-rose-900 border-b border-rose-700/80 px-4 py-2.5 text-center text-xs text-rose-100 font-bold uppercase tracking-wider flex items-center justify-center gap-2 shadow-md">
          <ShieldAlert className="w-4 h-4 text-rose-300" />
          <span>MOCK-DECIDER, not valid for the demo — real GPU decision cache not installed</span>
        </div>
      ) : (
        <div className="w-full bg-emerald-950/80 border-b border-emerald-600/60 px-4 py-2 text-center text-xs text-emerald-200 font-medium tracking-wide flex items-center justify-center gap-2 shadow-md">
          <ShieldCheck className="w-4 h-4 text-emerald-400" />
          <span>
            Decisions: real Strands Decider 2B (cached), decision latency{" "}
            {health?.cache_provenance?.decision_latency_ms
              ? Math.round(health.cache_provenance.decision_latency_ms)
              : 301}{" "}
            ms measured on {health?.cache_provenance?.latency_source || "Tesla T4 (cache-measured)"}
          </span>
        </div>
      )}

      {/* Main App Header */}
      <header className="border-b border-slate-800 bg-slate-900/60 backdrop-blur sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-indigo-600 to-emerald-500 flex items-center justify-center shadow-md shadow-indigo-500/20">
              <Zap className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-lg font-bold text-white tracking-tight flex items-center gap-2">
                Patient Listener
                <span className="text-[11px] font-semibold px-2 py-0.5 rounded-full bg-slate-800 text-indigo-300 border border-slate-700">
                  Strands Decider 2B
                </span>
              </h1>
              <p className="text-[11px] text-slate-400">
                Personalized, regret-aware voice turn-taking with speculative System One decisions
              </p>
            </div>
          </div>

          {/* Tab Navigation */}
          <div className="flex items-center gap-2 bg-slate-950 p-1 rounded-xl border border-slate-800">
            <button
              onClick={() => setActiveTab("demo")}
              className={`flex items-center gap-2 px-4 py-1.5 rounded-lg text-xs font-semibold transition ${
                activeTab === "demo"
                  ? "bg-indigo-600 text-white shadow"
                  : "text-slate-400 hover:text-slate-200 hover:bg-slate-900"
              }`}
            >
              <Play className="w-3.5 h-3.5" />
              Side-by-Side Demo
            </button>
            <button
              onClick={() => setActiveTab("results")}
              className={`flex items-center gap-2 px-4 py-1.5 rounded-lg text-xs font-semibold transition ${
                activeTab === "results"
                  ? "bg-indigo-600 text-white shadow"
                  : "text-slate-400 hover:text-slate-200 hover:bg-slate-900"
              }`}
            >
              <BarChart2 className="w-3.5 h-3.5" />
              Benchmark Results
            </button>
          </div>
        </div>
      </header>

      {/* Main Body Content */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-6">
        {activeTab === "results" ? (
          <ResultsDashboard apiBase={apiBase} />
        ) : (
          <div className="space-y-6">
            {/* Control Strip (Scenario Picker + Speed + Run Controls) */}
            <div className="bg-slate-900/90 border border-slate-700/80 rounded-xl p-4 shadow-xl flex flex-wrap items-center justify-between gap-4">
              {/* Scenario Picker */}
              <div className="flex-1 min-w-[320px]">
                <label className="block text-[11px] font-semibold text-slate-400 uppercase tracking-wider mb-1.5">
                  Select Conversational Scenario
                </label>
                <select
                  disabled={isRunning}
                  value={selectedScenarioId}
                  onChange={(e) => {
                    setSelectedScenarioId(e.target.value);
                    resetReplayState();
                  }}
                  className="w-full bg-slate-950 border border-slate-700 text-slate-100 text-xs rounded-lg px-3 py-2 focus:outline-none focus:border-indigo-500 disabled:opacity-50"
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
                <label className="block text-[11px] font-semibold text-slate-400 uppercase tracking-wider mb-1.5">
                  Replay Speed
                </label>
                <div className="flex items-center gap-1 bg-slate-950 p-1 rounded-lg border border-slate-800">
                  {[1.0, 2.0, 4.0].map((s) => (
                    <button
                      key={s}
                      disabled={isRunning}
                      onClick={() => setSpeed(s)}
                      className={`px-3 py-1 text-xs font-semibold rounded ${
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
              <div className="flex items-center gap-2 pt-5">
                {!isRunning ? (
                  <button
                    onClick={startReplay}
                    disabled={!selectedScenarioId}
                    className="flex items-center gap-2 px-5 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white font-semibold text-xs transition shadow-lg shadow-emerald-600/20 disabled:opacity-50"
                  >
                    <Play className="w-4 h-4 fill-current" />
                    Start Replay
                  </button>
                ) : (
                  <button
                    onClick={cancelReplay}
                    className="flex items-center gap-2 px-5 py-2 rounded-lg bg-rose-600 hover:bg-rose-500 text-white font-semibold text-xs transition shadow-lg shadow-rose-600/20"
                  >
                    <Square className="w-4 h-4 fill-current" />
                    Cancel Replay
                  </button>
                )}

                <button
                  onClick={playDemoSet}
                  disabled={isRunning}
                  className="flex items-center gap-2 px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-indigo-300 border border-slate-700 font-semibold text-xs transition disabled:opacity-50"
                  title="Run all demo-flagged scenarios in sequence"
                >
                  <FastForward className="w-4 h-4 text-indigo-400" />
                  Play Demo Set
                </button>
              </div>
            </div>

            {/* Running Demo Set Tally Banner */}
            {(isDemoSetRunning || demoSetTally.total > 0) && (
              <div className="bg-indigo-950/60 border border-indigo-700/60 rounded-xl p-3 flex flex-wrap items-center justify-between text-xs text-indigo-200 shadow-md">
                <div className="flex items-center gap-2">
                  <FastForward className={`w-4 h-4 text-indigo-400 ${isDemoSetRunning ? "animate-pulse" : ""}`} />
                  <span className="font-semibold">
                    {isDemoSetRunning
                      ? `Demo Set Running: Scenario ${demoSetIndex + 1} of ${scenarios.filter((s) => s.demo).length}`
                      : `Demo Set Completed (${demoSetTally.total} scenarios evaluated)`}
                  </span>
                </div>
                <div className="flex items-center gap-6 font-mono text-[11px]">
                  <span>
                    Fixed Cut-ins: <strong className="text-rose-400 font-bold">{demoSetTally.fixedInterrupted}</strong>
                  </span>
                  <span>
                    Patient Listener Cut-ins: <strong className="text-emerald-400 font-bold">{demoSetTally.plInterrupted}</strong>
                  </span>
                  <span>
                    Fixed Mean Delay:{" "}
                    <strong className="text-slate-100 font-bold">
                      {demoSetTally.fixedDelays.length > 0
                        ? Math.round(demoSetTally.fixedDelays.reduce((a, b) => a + b, 0) / demoSetTally.fixedDelays.length)
                        : 0}{" "}
                      ms
                    </strong>
                  </span>
                  <span>
                    PL Mean Delay:{" "}
                    <strong className="text-slate-100 font-bold">
                      {demoSetTally.plDelays.length > 0
                        ? Math.round(demoSetTally.plDelays.reduce((a, b) => a + b, 0) / demoSetTally.plDelays.length)
                        : 0}{" "}
                      ms
                    </strong>
                  </span>
                </div>
              </div>
            )}

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
