import React, { useEffect, useState, useRef } from "react";
import {
  Activity,
  AlertCircle,
  CheckCircle2,
  Clock,
  Mic,
  Radio,
  RefreshCw,
  Send,
  User,
  Zap,
} from "lucide-react";
import type { ScenarioItem, ServerMessage } from "./events";
import type { ConnectionStatus } from "./ws";
import { PatientListenerWSClient } from "./ws";

export const App: React.FC = () => {
  const [status, setStatus] = useState<ConnectionStatus>("disconnected");
  const [scenarios, setScenarios] = useState<ScenarioItem[]>([]);
  const [isLoadingScenarios, setIsLoadingScenarios] = useState<boolean>(false);
  const [scenariosError, setScenariosError] = useState<string | null>(null);
  const [messages, setMessages] = useState<ServerMessage[]>([]);
  const [selectedUser, setSelectedUser] = useState<string>("default_user");
  const [testInput, setTestInput] = useState<string>(
    "Can I get a refill on my blood pressure medication please?"
  );

  const wsClientRef = useRef<PatientListenerWSClient | null>(null);
  const apiBase = (import.meta.env.VITE_API_URL as string) || "http://localhost:8000";

  // Initialize WebSocket connection
  useEffect(() => {
    const client = new PatientListenerWSClient({
      userId: selectedUser,
      onStatusChange: (newStatus) => {
        setStatus(newStatus);
      },
      onMessage: (msg) => {
        setMessages((prev) => [msg, ...prev].slice(0, 50));
      },
      onError: (err) => {
        console.error("WS error:", err);
      },
    });

    wsClientRef.current = client;
    client.connect();

    return () => {
      client.disconnect();
    };
  }, [selectedUser]);

  // Fetch scenarios
  const fetchScenarios = async () => {
    setIsLoadingScenarios(true);
    setScenariosError(null);
    try {
      const resp = await fetch(`${apiBase}/scenarios`);
      if (!resp.ok) {
        throw new Error(`HTTP ${resp.status}: ${resp.statusText}`);
      }
      const data = await resp.json();
      setScenarios(data.scenarios || []);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      setScenariosError(msg);
    } finally {
      setIsLoadingScenarios(false);
    }
  };

  useEffect(() => {
    fetchScenarios();
  }, [apiBase]);

  const sendTestTurn = () => {
    if (!wsClientRef.current || status !== "connected") return;
    wsClientRef.current.send({ type: "SpeechStarted", timestamp_ms: 0.0 });
    wsClientRef.current.send({
      type: "SpeechDelta",
      text: testInput,
      timestamp_ms: 150.0,
    });
    wsClientRef.current.send({ type: "SpeechEnded", timestamp_ms: 2000.0 });
  };

  const statusConfig = {
    connected: {
      label: "Connected",
      color: "bg-emerald-500/20 text-emerald-400 border-emerald-500/40",
      dot: "bg-emerald-400",
      icon: CheckCircle2,
    },
    connecting: {
      label: "Connecting...",
      color: "bg-amber-500/20 text-amber-400 border-amber-500/40",
      dot: "bg-amber-400 animate-pulse",
      icon: RefreshCw,
    },
    reconnecting: {
      label: "Reconnecting...",
      color: "bg-amber-500/20 text-amber-400 border-amber-500/40",
      dot: "bg-amber-400 animate-pulse",
      icon: RefreshCw,
    },
    disconnected: {
      label: "Disconnected",
      color: "bg-gray-500/20 text-gray-400 border-gray-500/40",
      dot: "bg-gray-400",
      icon: AlertCircle,
    },
    error: {
      label: "Connection Error",
      color: "bg-rose-500/20 text-rose-400 border-rose-500/40",
      dot: "bg-rose-400",
      icon: AlertCircle,
    },
  }[status];

  const StatusIcon = statusConfig.icon;

  return (
    <div className="min-h-screen bg-[#090d16] text-gray-100 flex flex-col font-sans">
      {/* Top Navbar */}
      <header className="border-b border-gray-800/80 bg-[#0d1322]/80 backdrop-blur-md sticky top-0 z-50 px-6 py-4">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="p-2 rounded-xl bg-gradient-to-tr from-indigo-600 to-indigo-400 text-white shadow-lg shadow-indigo-500/20">
              <Mic className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <h1 className="text-lg font-bold tracking-tight text-white">
                  Patient Listener
                </h1>
                <span className="text-[10px] font-mono uppercase px-2 py-0.5 rounded-full bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">
                  Scaffold F1
                </span>
              </div>
              <p className="text-xs text-gray-400">
                Personalized, Regret-Aware Turn-Taking Architecture
              </p>
            </div>
          </div>

          <div className="flex items-center space-x-4">
            <div className="hidden sm:flex items-center space-x-2 text-xs font-mono text-gray-400 bg-gray-900/60 px-3 py-1.5 rounded-lg border border-gray-800">
              <span className="text-gray-500">API:</span>
              <span className="text-indigo-300">{apiBase}</span>
            </div>

            {/* Connection Status Badge */}
            <div
              id="connection-status-badge"
              className={`flex items-center space-x-2 px-3 py-1.5 rounded-lg border text-xs font-medium ${statusConfig.color}`}
            >
              <span className={`w-2 h-2 rounded-full ${statusConfig.dot}`} />
              <StatusIcon className="w-3.5 h-3.5" />
              <span>{statusConfig.label}</span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-6 grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: WS Interactive Sandbox & Event Stream */}
        <section className="lg:col-span-6 flex flex-col space-y-6">
          {/* Real-time Session Control Box */}
          <div className="bg-[#0f172a]/90 rounded-2xl border border-gray-800/80 p-5 shadow-xl">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center space-x-2">
                <Radio className="w-4 h-4 text-indigo-400" />
                <h2 className="text-sm font-semibold uppercase tracking-wider text-gray-200">
                  Live Turn Session
                </h2>
              </div>
              <div className="flex items-center space-x-2">
                <User className="w-3.5 h-3.5 text-gray-400" />
                <select
                  value={selectedUser}
                  onChange={(e) => setSelectedUser(e.target.value)}
                  className="bg-gray-900 border border-gray-700 text-xs rounded-lg px-2.5 py-1 text-gray-300 focus:outline-none focus:border-indigo-500"
                >
                  <option value="default_user">default_user (Neutral)</option>
                  <option value="user_fast_test_01">user_fast_test_01 (Fast Pauser)</option>
                  <option value="user_slow_test_01">user_slow_test_01 (Slow Pauser)</option>
                  <option value="user_filler_test_01">user_filler_test_01 (Filler Heavy)</option>
                </select>
              </div>
            </div>

            <div className="space-y-3">
              <div>
                <label className="block text-xs font-medium text-gray-400 mb-1">
                  Test Utterance Fragment
                </label>
                <div className="flex space-x-2">
                  <input
                    type="text"
                    value={testInput}
                    onChange={(e) => setTestInput(e.target.value)}
                    placeholder="Speak or type test phrase..."
                    className="flex-1 bg-gray-900/90 border border-gray-700/80 rounded-xl px-3.5 py-2 text-sm text-gray-200 placeholder-gray-500 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition-all"
                  />
                  <button
                    id="btn-send-test-turn"
                    onClick={sendTestTurn}
                    disabled={status !== "connected"}
                    className="flex items-center space-x-1.5 px-4 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 active:bg-indigo-700 disabled:opacity-40 disabled:cursor-not-allowed text-white text-xs font-semibold shadow-md shadow-indigo-600/20 transition-all"
                  >
                    <Send className="w-3.5 h-3.5" />
                    <span>Send Turn</span>
                  </button>
                </div>
              </div>
              <p className="text-[11px] text-gray-400">
                Dispatches <code className="text-indigo-300">SpeechStarted</code>, <code className="text-indigo-300">SpeechDelta</code>, and <code className="text-indigo-300">SpeechEnded</code> over WebSocket to exercise the server policy.
              </p>
            </div>
          </div>

          {/* Event Stream Log */}
          <div className="flex-1 bg-[#0f172a]/90 rounded-2xl border border-gray-800/80 p-5 shadow-xl flex flex-col min-h-[380px]">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center space-x-2">
                <Activity className="w-4 h-4 text-emerald-400" />
                <h2 className="text-sm font-semibold uppercase tracking-wider text-gray-200">
                  WebSocket Event Log
                </h2>
              </div>
              <span className="text-xs font-mono text-gray-500">
                {messages.length} events
              </span>
            </div>

            <div
              id="event-log-container"
              className="flex-1 overflow-y-auto space-y-2 pr-1 max-h-[420px] font-mono text-xs"
            >
              {messages.length === 0 ? (
                <div className="h-full flex flex-col items-center justify-center text-gray-500 text-center py-16">
                  <Activity className="w-8 h-8 text-gray-600 mb-2 stroke-1" />
                  <p>No WebSocket events received yet.</p>
                  <p className="text-[11px] text-gray-600 mt-1">
                    Connect backend and send a test turn above.
                  </p>
                </div>
              ) : (
                messages.map((msg, idx) => (
                  <div
                    key={idx}
                    className="p-2.5 rounded-lg bg-gray-900/80 border border-gray-800/90 hover:border-gray-700/80 transition-all"
                  >
                    <div className="flex items-center justify-between mb-1">
                      <span className="font-semibold text-indigo-400 uppercase text-[11px]">
                        {msg.type}
                      </span>
                      {"timestamp_ms" in msg && (
                        <span className="text-[10px] text-gray-500">
                          {Number(msg.timestamp_ms).toFixed(0)} ms
                        </span>
                      )}
                    </div>
                    {msg.type === "decision" && (
                      <div className="text-gray-300 text-[11px] space-y-0.5">
                        <div className="flex justify-between">
                          <span>Action:</span>
                          <span
                            className={`font-bold ${
                              msg.action === "RESPOND"
                                ? "text-emerald-400"
                                : "text-amber-400"
                            }`}
                          >
                            {msg.action}
                          </span>
                        </div>
                        <div className="flex justify-between text-gray-400">
                          <span>Checkpoint:</span>
                          <span>{msg.checkpoint} ms</span>
                        </div>
                        <div className="flex justify-between text-gray-400">
                          <span>p_done:</span>
                          <span>{(msg.p_done * 100).toFixed(1)}%</span>
                        </div>
                      </div>
                    )}
                    {msg.type === "reply_chunk" && (
                      <div className="text-emerald-300 text-xs mt-1">
                        &ldquo;{msg.chunk}&rdquo;
                      </div>
                    )}
                    {msg.type === "reply_done" && (
                      <div className="text-emerald-400 font-sans text-xs mt-1 bg-emerald-950/40 p-2 rounded border border-emerald-800/50">
                        {msg.full_reply}
                      </div>
                    )}
                    {msg.type !== "decision" &&
                      msg.type !== "reply_chunk" &&
                      msg.type !== "reply_done" && (
                        <pre className="text-[10px] text-gray-400 overflow-x-auto whitespace-pre-wrap">
                          {JSON.stringify(msg, null, 2)}
                        </pre>
                      )}
                  </div>
                ))
              )}
            </div>
          </div>
        </section>

        {/* Right Column: Scenario Explorer (GET /scenarios) */}
        <section className="lg:col-span-6 flex flex-col space-y-6">
          <div className="bg-[#0f172a]/90 rounded-2xl border border-gray-800/80 p-5 shadow-xl flex-1 flex flex-col">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center space-x-2">
                <Zap className="w-4 h-4 text-amber-400" />
                <h2 className="text-sm font-semibold uppercase tracking-wider text-gray-200">
                  Scenario Catalog
                </h2>
              </div>
              <button
                id="btn-refresh-scenarios"
                onClick={fetchScenarios}
                disabled={isLoadingScenarios}
                className="flex items-center space-x-1 px-2.5 py-1 rounded-lg bg-gray-800 hover:bg-gray-700 text-gray-300 text-xs transition-all disabled:opacity-50"
              >
                <RefreshCw
                  className={`w-3 h-3 ${isLoadingScenarios ? "animate-spin" : ""}`}
                />
                <span>Refresh</span>
              </button>
            </div>

            {scenariosError ? (
              <div className="p-4 rounded-xl bg-rose-950/40 border border-rose-800/60 text-rose-300 text-xs flex items-start space-x-3">
                <AlertCircle className="w-5 h-5 flex-shrink-0 text-rose-400" />
                <div>
                  <p className="font-semibold">Failed to load scenarios</p>
                  <p className="text-[11px] text-rose-400 mt-0.5">{scenariosError}</p>
                </div>
              </div>
            ) : isLoadingScenarios ? (
              <div className="flex-1 flex flex-col items-center justify-center text-gray-500 py-16">
                <RefreshCw className="w-6 h-6 animate-spin mb-2" />
                <p className="text-xs">Fetching test scenarios...</p>
              </div>
            ) : scenarios.length === 0 ? (
              <div className="flex-1 flex flex-col items-center justify-center text-gray-500 py-16 text-center">
                <p className="text-xs">No scenarios available.</p>
              </div>
            ) : (
              <div
                id="scenarios-list"
                className="flex-1 overflow-y-auto space-y-2.5 pr-1 max-h-[560px]"
              >
                {scenarios.map((sc) => (
                  <div
                    key={sc.id}
                    className="p-3.5 rounded-xl bg-gray-900/60 border border-gray-800 hover:border-indigo-500/40 transition-all group"
                  >
                    <div className="flex items-center justify-between mb-1.5">
                      <div className="flex items-center space-x-2">
                        <span className="font-mono text-xs font-semibold text-indigo-300">
                          {sc.id}
                        </span>
                        <span
                          className={`text-[10px] uppercase font-mono px-2 py-0.5 rounded-full border ${
                            sc.archetype === "fast_pauser"
                              ? "bg-sky-500/10 text-sky-400 border-sky-500/30"
                              : sc.archetype === "slow_pauser"
                              ? "bg-amber-500/10 text-amber-400 border-amber-500/30"
                              : "bg-purple-500/10 text-purple-400 border-purple-500/30"
                          }`}
                        >
                          {sc.archetype.replace("_", " ")}
                        </span>
                      </div>
                      <div className="flex items-center space-x-1 text-[11px] text-gray-400">
                        <Clock className="w-3 h-3 text-gray-500" />
                        <span>{sc.true_end_time_ms} ms</span>
                      </div>
                    </div>

                    <p className="text-xs text-gray-300 italic group-hover:text-white transition-colors line-clamp-2">
                      &ldquo;{sc.text_preview}&rdquo;
                    </p>

                    <div className="flex items-center justify-between mt-2 pt-2 border-t border-gray-800/60 text-[10px] text-gray-400">
                      <span>User: <strong className="text-gray-300">{sc.user_id}</strong></span>
                      <span>Category: <strong className="text-gray-300">{sc.category}</strong></span>
                      <span>
                        Turn:{" "}
                        <strong
                          className={
                            sc.is_turn_complete
                              ? "text-emerald-400"
                              : "text-amber-400"
                          }
                        >
                          {sc.is_turn_complete ? "Complete" : "Mid-thought"}
                        </strong>
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </section>
      </main>
    </div>
  );
};

export default App;
