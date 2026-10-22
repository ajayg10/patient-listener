import React from "react";
import { Clock, AlertTriangle, CheckCircle2 } from "lucide-react";
import { formatResultLine } from "./resultLine";

export interface SpeechSegment {
  startMs: number;
  endMs: number;
  text?: string;
}

export interface EngineResult {
  respondedAtMs: number | null;
  interrupted: boolean | null;
  responseDelayMs: number | null;
}

interface ReplayTimelineProps {
  speechSegments: SpeechSegment[];
  trueEndTimeMs: number;
  currentTimeMs: number;
  fixedResult: EngineResult;
  plResult: EngineResult;
  streamError?: string | null;
}

export const ReplayTimeline: React.FC<ReplayTimelineProps> = ({
  speechSegments,
  trueEndTimeMs,
  currentTimeMs,
  fixedResult,
  plResult,
  streamError,
}) => {
  // Determine timeline boundary
  const allTimes = [
    trueEndTimeMs,
    currentTimeMs,
    fixedResult.respondedAtMs ?? 0,
    plResult.respondedAtMs ?? 0,
    ...speechSegments.map((s) => s.endMs),
  ];
  const maxTime = Math.max(...allTimes, 1000) + 400;

  const toPercent = (ms: number): number => {
    return Math.min(100, Math.max(0, (ms / maxTime) * 100));
  };

  return (
    <div className="w-full bg-slate-900/90 border border-slate-700/80 rounded-xl p-4 shadow-lg">
      <div className="flex items-center justify-between mb-3 text-xs text-slate-300">
        <div className="flex items-center gap-2 font-medium">
          <Clock className="w-4 h-4 text-indigo-400" />
          <span>Unified Conversational Timeline (0 ms – {Math.round(maxTime)} ms)</span>
        </div>
        <div className="flex items-center gap-4 text-[11px]">
          <span className="flex items-center gap-1.5">
            <span className="w-3 h-2 rounded-sm bg-indigo-500 inline-block" /> User Speaking
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-3 h-2 rounded-sm bg-slate-800 border border-slate-600 inline-block" /> Pause Silence
          </span>
          <span className="flex items-center gap-1.5 text-rose-400">
            <span className="w-2.5 h-2.5 rounded-full bg-rose-500 inline-block" /> Fixed Engine Cut-In
          </span>
          <span className="flex items-center gap-1.5 text-emerald-400">
            <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 inline-block" /> Patient Listener
          </span>
        </div>
      </div>

      {/* Main Track */}
      <div className="relative w-full h-14 bg-slate-950/80 border border-slate-800 rounded-lg overflow-hidden my-2">
        {/* Background grid lines every 1000ms */}
        {Array.from({ length: Math.ceil(maxTime / 1000) }).map((_, idx) => {
          const tMs = idx * 1000;
          const leftPct = toPercent(tMs);
          return (
            <div
              key={idx}
              className="absolute top-0 bottom-0 border-l border-slate-800/60 text-[9px] text-slate-600 pl-1 pt-0.5 select-none"
              style={{ left: `${leftPct}%` }}
            >
              {tMs}ms
            </div>
          );
        })}

        {/* User Speaking Segments */}
        {speechSegments.map((seg, idx) => {
          const left = toPercent(seg.startMs);
          const width = Math.max(0.8, toPercent(seg.endMs) - left);
          return (
            <div
              key={idx}
              className="absolute top-2.5 h-4 bg-indigo-500/80 hover:bg-indigo-400 rounded-sm transition-all border border-indigo-400/40"
              style={{ left: `${left}%`, width: `${width}%` }}
              title={`Speech: ${Math.round(seg.startMs)}ms - ${Math.round(seg.endMs)}ms`}
            />
          );
        })}

        {/* True End Marker (Dashed flag line) */}
        {trueEndTimeMs > 0 && (
          <div
            className="absolute top-0 bottom-0 border-r-2 border-dashed border-amber-400 z-10"
            style={{ left: `${toPercent(trueEndTimeMs)}%` }}
          >
            <span className="absolute -top-1 -right-2 px-1.5 py-0.5 bg-amber-400 text-slate-950 font-bold text-[9px] rounded uppercase shadow-sm">
              True End ({Math.round(trueEndTimeMs)}ms)
            </span>
          </div>
        )}

        {/* Fixed Engine Marker */}
        {fixedResult.respondedAtMs !== null && (
          <div
            className="absolute top-7 z-20 transition-all flex flex-col items-center"
            style={{ left: `${toPercent(fixedResult.respondedAtMs)}%` }}
          >
            <div
              className={`w-3.5 h-3.5 rounded-full border-2 shadow-md ${
                fixedResult.interrupted
                  ? "bg-rose-500 border-white animate-pulse"
                  : "bg-amber-500 border-slate-950"
              }`}
            />
            <span
              className={`text-[9px] font-semibold whitespace-nowrap px-1 py-0.5 rounded shadow mt-0.5 ${
                fixedResult.interrupted
                  ? "bg-rose-950 text-rose-300 border border-rose-600"
                  : "bg-slate-800 text-amber-300 border border-slate-600"
              }`}
            >
              Fixed {Math.round(fixedResult.respondedAtMs)}ms
            </span>
          </div>
        )}

        {/* Patient Listener Engine Marker */}
        {plResult.respondedAtMs !== null && (
          <div
            className="absolute top-0 z-20 transition-all flex flex-col items-center"
            style={{ left: `${toPercent(plResult.respondedAtMs)}%` }}
          >
            <div
              className={`w-3.5 h-3.5 rounded-full border-2 shadow-md ${
                plResult.interrupted
                  ? "bg-rose-500 border-white"
                  : "bg-emerald-400 border-slate-950"
              }`}
            />
            <span
              className={`text-[9px] font-semibold whitespace-nowrap px-1 py-0.5 rounded shadow ${
                plResult.interrupted
                  ? "bg-rose-950 text-rose-300 border border-rose-600"
                  : "bg-emerald-950 text-emerald-300 border border-emerald-600"
              }`}
            >
              Decider {Math.round(plResult.respondedAtMs)}ms
            </span>
          </div>
        )}

        {/* Live Playhead Marker */}
        {currentTimeMs > 0 && (
          <div
            className="absolute top-0 bottom-0 w-0.5 bg-cyan-400 shadow-[0_0_8px_rgba(34,211,238,0.8)] z-30"
            style={{ left: `${toPercent(currentTimeMs)}%` }}
          >
            <div className="w-2 h-2 -ml-[3px] bg-cyan-400 rotate-45" />
          </div>
        )}
      </div>

      {/* Outcome Verdict Summary */}
      {(fixedResult.respondedAtMs !== null || plResult.respondedAtMs !== null) && (
        <div className="mt-3 pt-2.5 border-t border-slate-800 flex flex-wrap items-center justify-between gap-3 text-xs">
          <div className="flex items-center gap-2">
            {fixedResult.interrupted ? (
              <span className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-rose-500/20 text-rose-300 border border-rose-500/40 font-medium">
                <AlertTriangle className="w-3.5 h-3.5 text-rose-400" />
                Fixed Engine cut in prematurely at {Math.round(fixedResult.respondedAtMs ?? 0)} ms
              </span>
            ) : fixedResult.respondedAtMs !== null ? (
              <span className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-slate-800 text-slate-300 border border-slate-700">
                Fixed answered after {Math.round(fixedResult.responseDelayMs ?? 0)} ms delay
              </span>
            ) : null}

            {(() => {
              const resLine = formatResultLine({
                interrupted: plResult.interrupted,
                responseDelayMs: plResult.responseDelayMs,
                respondedAtMs: plResult.respondedAtMs,
                hasResponse: plResult.respondedAtMs !== null,
                streamError: streamError,
              });
              const isSuccess = plResult.respondedAtMs !== null && plResult.interrupted === false && !streamError;
              const isError = Boolean(streamError);
              const isCutIn = plResult.interrupted === true;

              return (
                <span
                  className={`flex items-center gap-1.5 px-2.5 py-1 rounded font-medium ${
                    isSuccess
                      ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/40"
                      : isCutIn || isError
                      ? "bg-rose-500/20 text-rose-300 border border-rose-500/40"
                      : "bg-slate-800 text-slate-400 border border-slate-700"
                  }`}
                >
                  {isSuccess ? (
                    <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                  ) : (
                    <AlertTriangle className="w-3.5 h-3.5 text-rose-400" />
                  )}
                  {resLine}
                </span>
              );
            })()}
          </div>

          <div className="text-slate-400 text-[11px]">
            {fixedResult.interrupted && !plResult.interrupted && !streamError && plResult.respondedAtMs !== null && (
              <span className="text-emerald-400 font-semibold">
                ✓ Patient Listener successfully prevented premature cutoff.
              </span>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
