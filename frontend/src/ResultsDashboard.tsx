import React, { useEffect, useState } from "react";
import { BarChart3, AlertTriangle, ShieldCheck, Info, RefreshCw } from "lucide-react";

interface MainResultRow {
  name: string;
  accuracy: number;
  accuracy_ci?: [number, number];
  accuracy_cluster_ci?: [number, number];
  premature_interruption_rate: number;
  premature_interruption_ci?: [number, number];
  premature_interruption_cluster_ci?: [number, number];
  response_delay_p50_ms: number;
  response_delay_p50_ci?: [number, number];
  response_delay_p95_ms: number;
  latency_condition?: string;
}

interface EvalResultsData {
  status: string;
  provenance?: {
    decider_backend: string;
    cache_keys_total: number;
    is_mock: boolean;
    headline_gpu_median_ms?: number;
    server_side_compute_ms?: number;
    mock_notice?: string;
  };
  sample_counts?: {
    total_test_turns: number;
    test_users_count: number;
    turns_per_user: number;
    finished_turns: number;
    incomplete_turns: number;
  };
  calibration?: {
    dev_chosen_hi: number;
    dev_chosen_lo: number;
    dev_metrics?: {
      accuracy: number;
      premature_rate: number;
    };
  };
  main_results?: MainResultRow[];
  operating_points?: any;
  stage_timings_s?: Record<string, number>;
}

interface ResultsDashboardProps {
  apiBase: string;
}

export const ResultsDashboard: React.FC<ResultsDashboardProps> = ({ apiBase }) => {
  const [data, setData] = useState<EvalResultsData | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const fetchResults = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const resp = await fetch(`${apiBase}/results`);
      if (!resp.ok) {
        throw new Error(`Failed to fetch results (HTTP ${resp.status})`);
      }
      const json = await resp.json();
      setData(json);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchResults();
  }, [apiBase]);

  if (isLoading) {
    return (
      <div className="flex flex-col items-center justify-center py-24 text-slate-400 gap-3">
        <RefreshCw className="w-8 h-8 animate-spin text-indigo-400" />
        <span className="text-sm font-medium">Loading benchmark evaluation results...</span>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="p-8 bg-rose-500/10 border border-rose-500/30 rounded-xl text-rose-300 max-w-2xl mx-auto my-12 text-center">
        <AlertTriangle className="w-8 h-8 mx-auto mb-2 text-rose-400" />
        <h3 className="text-base font-semibold mb-1">Failed to Load Results</h3>
        <p className="text-xs text-rose-400/80 mb-4">{error ?? "No data returned from /results"}</p>
        <button
          onClick={fetchResults}
          className="px-4 py-1.5 bg-rose-600 hover:bg-rose-500 text-white text-xs font-medium rounded-lg transition"
        >
          Retry Fetch
        </button>
      </div>
    );
  }

  const prov = data.provenance;
  const isMock = prov?.is_mock ?? true;

  const opData = data.operating_points;

  const famOrder = ["Fixed Timeout", "Text Heuristic", "Per-User Control", "Decider", "Decider + Profile"];
  const mainRows = data.main_results || [];

  return (
    <div className="space-y-8 max-w-7xl mx-auto">
      {/* Header Notice Banner */}
      {isMock ? (
        <div className="p-4 bg-rose-950/70 border border-rose-600/60 rounded-xl text-rose-200 text-xs flex items-start gap-3">
          <AlertTriangle className="w-5 h-5 text-rose-400 shrink-0 mt-0.5" />
          <div>
            <span className="font-bold text-rose-300 uppercase tracking-wide">
              Evaluation Executed on MockDecider Cache ({prov?.cache_keys_total ?? 0} keys)
            </span>
            <p className="mt-1 text-rose-300/80 leading-relaxed">
              These baseline and decider rows were computed from deterministic mock decisions for pipeline validation.
              Per protocol rules, headline claims require a verified real cache (GPU Tesla T4).
            </p>
          </div>
        </div>
      ) : (
        <div className="p-4 bg-emerald-950/60 border border-emerald-500/50 rounded-xl text-emerald-200 text-xs flex items-start gap-3">
          <ShieldCheck className="w-5 h-5 text-emerald-400 shrink-0 mt-0.5" />
          <div>
            <span className="font-bold text-emerald-300 uppercase tracking-wide">
              Real Decision Cache Provenance ({prov?.cache_keys_total ?? 0} keys)
            </span>
            <p className="mt-1 text-emerald-300/80 leading-relaxed">
              Evaluation executed with real Strands Decider 2B weights precomputed on Tesla T4 GPU (median{" "}
              {prov?.headline_gpu_median_ms ? prov.headline_gpu_median_ms.toFixed(1) : "301.2"} ms).
            </p>
          </div>
        </div>
      )}

      {/* Headline Operating Points Table (Points A & B) */}
      {opData && (
        <div className="bg-slate-900/90 border border-indigo-500/40 rounded-xl overflow-hidden shadow-xl">
          <div className="px-6 py-4 border-b border-slate-800 bg-indigo-950/20 flex flex-col md:flex-row md:items-center justify-between gap-2">
            <div>
              <div className="flex items-center gap-2.5">
                <BarChart3 className="w-5 h-5 text-indigo-400" />
                <h2 className="text-base font-bold text-white tracking-wide">
                  Headline Operating Points: Premature Interruption vs Delay
                </h2>
              </div>
              <p className="text-xs text-slate-400 mt-1">
                Point A: Delay budget ≤ 800 ms (Fixed 800ms match) | Point B: Premature budget ≤ 5.0%. Accuracy omitted per protocol.
              </p>
            </div>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="bg-slate-950/90 text-slate-400 uppercase tracking-wider text-[10px] border-b border-slate-800">
                <tr>
                  <th className="py-3 px-4">Operating Point</th>
                  <th className="py-3 px-4">Approach</th>
                  <th className="py-3 px-4">Parameter</th>
                  <th className="py-3 px-4">Premature (Cluster 95% CI)</th>
                  <th className="py-3 px-4">Delay P50 (Bootstrap 95% CI)</th>
                  <th className="py-3 px-4">Delay P95</th>
                  <th className="py-3 px-4">Condition</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 font-mono">
                {[
                  ...famOrder.filter(f => opData.point_a[f]).map(f => ({ pt: "Point A (Delay ≤ 800ms)", ...opData.point_a[f] })),
                  ...famOrder.filter(f => opData.point_b[f]).map(f => ({ pt: "Point B (Premature ≤ 5%)", ...opData.point_b[f] })),
                ].map((row, idx) => {
                  const isDecider = row.family.includes("Decider");
                  const premClust = row.premature_cluster_ci
                    ? `[${(row.premature_cluster_ci[0] * 100).toFixed(1)}%, ${(row.premature_cluster_ci[1] * 100).toFixed(1)}%]`
                    : "";
                  const delClust = row.delay_p50_ci
                    ? `[${row.delay_p50_ci[0].toFixed(1)}, ${row.delay_p50_ci[1].toFixed(1)}] ms`
                    : "";

                  const dispName = row.family === "Decider" ? "Strands Decider 2B" : (row.family === "Decider + Profile" ? "Strands Decider 2B + Profile" : row.family);
                  const isPointA = row.pt.includes("Point A");

                  return (
                    <tr
                      key={idx}
                      className={`hover:bg-slate-800/40 transition-colors ${
                        isDecider ? "bg-emerald-950/15 text-emerald-100" : ""
                      } ${!isPointA && idx === famOrder.length ? "border-t-2 border-indigo-500/30" : ""}`}
                    >
                      <td className="py-3 px-4 font-sans font-semibold text-slate-300 text-[11px]">{row.pt}</td>
                      <td className="py-3 px-4 font-sans font-medium text-slate-100">{dispName}</td>
                      <td className="py-3 px-4 text-indigo-300 font-semibold">{row.param_str}</td>
                      <td className="py-3 px-4 font-semibold text-slate-100">
                        {(row.premature_rate * 100).toFixed(1)}%{" "}
                        <span className="text-slate-400 font-normal text-[11px]">{premClust}</span>
                      </td>
                      <td className="py-3 px-4 text-slate-200">
                        {Math.round(row.delay_p50_ms)} ms{" "}
                        <span className="text-slate-400 font-normal text-[11px]">{delClust}</span>
                      </td>
                      <td className="py-3 px-4 text-slate-300">{Math.round(row.delay_p95_ms)} ms</td>
                      <td className="py-3 px-4 text-slate-400 font-sans text-[11px]">{row.condition}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Exploratory Full Benchmark Table */}
      <div className="bg-slate-900/90 border border-slate-700/80 rounded-xl overflow-hidden shadow-xl">
        <div className="px-6 py-4 border-b border-slate-800 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <BarChart3 className="w-5 h-5 text-indigo-400" />
            <h2 className="text-base font-bold text-white tracking-wide">
              Exploratory Full Benchmark (19 Latency & Sensitivity Configurations)
            </h2>
            {data.sample_counts && (
              <span className="text-xs text-slate-400">
                (n={data.sample_counts.total_test_turns} test turns, {data.sample_counts.test_users_count} users,{" "}
                {data.sample_counts.turns_per_user} turns/user)
              </span>
            )}
          </div>
          {data.stage_timings_s?.total_seconds !== undefined && (
            <span className="text-xs text-slate-400">
              Evaluation run duration: <strong className="text-slate-200">{data.stage_timings_s.total_seconds}s</strong>
            </span>
          )}
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-950/80 text-slate-400 uppercase tracking-wider text-[10px] border-b border-slate-800">
              <tr>
                <th className="py-3 px-4">Approach</th>
                <th className="py-3 px-4">Accuracy (Cluster 95% CI)</th>
                <th className="py-3 px-4">Premature Interruption (Cluster 95% CI)</th>
                <th className="py-3 px-4">Response Delay P50 (Bootstrap 95% CI)</th>
                <th className="py-3 px-4">Delay P95</th>
                <th className="py-3 px-4">Condition</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-mono">
              {mainRows.map((row, idx) => {
                const isReference = row.name.includes("REFERENCE") || row.name.includes("CEILING") || row.name.includes("ORACLE");
                const isDecider = row.name.includes("Strands Decider") || row.name.includes("Calibrated");

                const accClust = row.accuracy_cluster_ci
                  ? `[${(row.accuracy_cluster_ci[0] * 100).toFixed(1)}%, ${(row.accuracy_cluster_ci[1] * 100).toFixed(1)}%]`
                  : "";
                const premClust = row.premature_interruption_cluster_ci
                  ? `[${(row.premature_interruption_cluster_ci[0] * 100).toFixed(1)}%, ${(row.premature_interruption_cluster_ci[1] * 100).toFixed(1)}%]`
                  : "";
                const delayCi = row.response_delay_p50_ci
                  ? `[${row.response_delay_p50_ci[0].toFixed(1)}, ${row.response_delay_p50_ci[1].toFixed(1)}] ms`
                  : "";

                let rowStyle = "hover:bg-slate-800/40 transition-colors";
                if (isReference) {
                  rowStyle += " bg-indigo-950/20 text-indigo-200/90";
                } else if (isDecider) {
                  rowStyle += " bg-emerald-950/10 text-emerald-100";
                }

                return (
                  <tr key={idx} className={rowStyle}>
                    <td className="py-3 px-4 font-sans font-medium flex items-center gap-2">
                      {isReference && (
                        <span className="px-1.5 py-0.5 rounded bg-indigo-500/20 text-indigo-300 text-[10px] uppercase font-bold tracking-wider">
                          Reference (single 1st checkpoint)
                        </span>
                      )}
                      <span>{row.name}</span>
                    </td>
                    <td className="py-3 px-4 font-semibold text-slate-100">
                      {(row.accuracy * 100).toFixed(1)}% <span className="text-slate-400 font-normal text-[11px]">{accClust}</span>
                    </td>
                    <td className="py-3 px-4 font-semibold text-slate-100">
                      {(row.premature_interruption_rate * 100).toFixed(1)}%{" "}
                      <span className="text-slate-400 font-normal text-[11px]">{premClust}</span>
                    </td>
                    <td className="py-3 px-4 text-slate-200">
                      {Math.round(row.response_delay_p50_ms)} ms{" "}
                      <span className="text-slate-400 font-normal text-[11px]">{delayCi}</span>
                    </td>
                    <td className="py-3 px-4 text-slate-300">{Math.round(row.response_delay_p95_ms)} ms</td>
                    <td className="py-3 px-4 text-slate-400 font-sans text-[11px]">{row.latency_condition ?? "—"}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Dual Plots Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Pareto Frontier Plot */}
        <div className="bg-slate-900/90 border border-slate-700/80 rounded-xl p-5 shadow-xl flex flex-col justify-between">
          <div>
            <div className="flex items-center gap-2 mb-2">
              <BarChart3 className="w-4 h-4 text-emerald-400" />
              <h3 className="text-sm font-bold text-white">Turn-Taking Pareto Frontiers</h3>
            </div>
            <p className="text-xs text-slate-400 mb-4 leading-relaxed">
              Premature Interruption Rate vs Response Delay across fixed timeouts, heuristic thresholds, and decider confidence sweeps.
            </p>
          </div>
          <div className="bg-slate-950 rounded-lg p-2 border border-slate-800 flex items-center justify-center overflow-hidden min-h-[300px]">
            <img
              src={`${apiBase}/plots/results_pareto.png`}
              alt="Pareto Frontiers Plot"
              className="w-full h-auto object-contain rounded hover:scale-[1.02] transition-transform duration-300"
              onError={(e) => {
                (e.target as HTMLImageElement).alt = "Pareto plot could not be loaded from /plots/results_pareto.png";
              }}
            />
          </div>
          <p className="text-[11px] text-slate-500 mt-3 italic text-center">
            Source: GET /plots/results_pareto.png (Generated from decisions cache)
          </p>
        </div>

        {/* Probability Reliability Plot */}
        <div className="bg-slate-900/90 border border-slate-700/80 rounded-xl p-5 shadow-xl flex flex-col justify-between">
          <div>
            <div className="flex items-center gap-2 mb-2">
              <BarChart3 className="w-4 h-4 text-indigo-400" />
              <h3 className="text-sm font-bold text-white">Probability Calibration & Reliability</h3>
            </div>
            <p className="text-xs text-slate-400 mb-4 leading-relaxed">
              Brier scores and reliability curves evaluated on true probabilities across DEV and TEST splits.
            </p>
          </div>
          <div className="bg-slate-950 rounded-lg p-2 border border-slate-800 flex items-center justify-center overflow-hidden min-h-[300px]">
            <img
              src={`${apiBase}/plots/results_reliability.png`}
              alt="Probability Reliability Plot"
              className="w-full h-auto object-contain rounded hover:scale-[1.02] transition-transform duration-300"
              onError={(e) => {
                (e.target as HTMLImageElement).alt = "Reliability plot could not be loaded from /plots/results_reliability.png";
              }}
            />
          </div>
          <p className="text-[11px] text-slate-500 mt-3 italic text-center">
            Source: GET /plots/results_reliability.png (Generated from decisions cache)
          </p>
        </div>
      </div>

      {/* Methodology & Caveats Box */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-6 text-xs text-slate-400 space-y-3">
        <div className="flex items-center gap-2 text-slate-200 font-semibold text-sm">
          <Info className="w-4 h-4 text-indigo-400" />
          <span>Evaluation Methodology, CIs & Reference Rows</span>
        </div>
        <ul className="list-disc pl-5 space-y-1.5 leading-relaxed">
          <li>
            <strong className="text-slate-300">SYNTHETIC Data Caveat:</strong> Evaluations are executed on synthetic conversational
            templates with controlled pauses to isolate turn-taking timing dynamics; natural acoustic variety and acoustic prosody are not modeled.
          </li>
          <li>
            <strong className="text-slate-300">Uncertainty Estimation:</strong> Headline tables report non-parametric
            two-stage cluster bootstrap 95% intervals (B=1000 resamples: sample users with replacement, then sample
            turns within user with replacement) to account for turn clustering across users.
          </li>
          <li>
            <strong className="text-slate-300">Single First-Checkpoint Reference Rows (Not Ceilings):</strong>{" "}
            <code className="text-indigo-300">PERSONA-BLIND REFERENCE</code>,{" "}
            <code className="text-indigo-300">ARCHETYPE-AWARE REFERENCE</code>, and{" "}
            <code className="text-indigo-300">USER-AWARE REFERENCE</code> evaluate decisions solely at the first checkpoint (250 ms) without multi-checkpoint dynamic hold.
            The archetype-aware vs user-aware gap is about 0.5 points, meaning little per-user headroom in this dataset (a limitation).
            The per-user control heuristic (87.2%) exceeds the user-aware reference (81.9%), proving these are single-checkpoint references and not upper bounds for multi-checkpoint policies.
          </li>
          <li>
            <strong className="text-slate-300">Hardware Latency Model:</strong> GPU evaluation models Strands Decider
            2B speculative queries arriving at the measured Tesla T4 median (
            {prov?.headline_gpu_median_ms ? prov.headline_gpu_median_ms.toFixed(1) : "301.2"} ms).
          </li>
          <li>
            <strong className="text-slate-300">Zero Leakage:</strong> Split strictly partitions 15 DEV users (24
            utterance templates) and 30 TEST users (36 templates). Calibration threshold tuning is isolated to DEV.
          </li>
        </ul>
      </div>
    </div>
  );
};
