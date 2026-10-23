// Pure tally logic for the "Play demo set" runner.
// One ScenarioOutcome per scenario, built from that scenario's own final events,
// so cut-ins can never exceed the number of scenarios evaluated.

export interface EngineOutcome {
  interrupted: boolean;
  respondedAtMs: number | null;
  responseDelayMs: number | null;
}

export interface ScenarioOutcome {
  scenarioId: string;
  trueEndMs: number;
  fixed: EngineOutcome | null;
  patientListener: EngineOutcome | null;
  error: string | null;
}

export interface EngineSummary {
  cutIns: number;
  // Answers that came before the user finished: mean ms early (true end minus response time).
  earlyCount: number;
  meanEarlyMs: number | null;
  // Answers after the true end: mean response delay.
  lateCount: number;
  meanDelayMs: number | null;
  noAnswer: number;
}

export interface DemoTally {
  evaluated: number;
  errored: number;
  fixed: EngineSummary;
  patientListener: EngineSummary;
}

const mean = (xs: number[]): number | null =>
  xs.length === 0 ? null : Math.round(xs.reduce((a, b) => a + b, 0) / xs.length);

function summarizeEngine(
  outcomes: ScenarioOutcome[],
  pick: (o: ScenarioOutcome) => EngineOutcome | null,
): EngineSummary {
  let cutIns = 0;
  let noAnswer = 0;
  const early: number[] = [];
  const late: number[] = [];
  for (const o of outcomes) {
    const e = pick(o);
    if (!e || e.respondedAtMs === null) {
      noAnswer += 1;
      continue;
    }
    if (e.interrupted) {
      cutIns += 1;
      early.push(Math.max(0, o.trueEndMs - e.respondedAtMs));
    } else if (e.responseDelayMs !== null) {
      late.push(e.responseDelayMs);
    }
  }
  return {
    cutIns,
    earlyCount: early.length,
    meanEarlyMs: mean(early),
    lateCount: late.length,
    meanDelayMs: mean(late),
    noAnswer,
  };
}

export function summarizeDemoSet(outcomes: ScenarioOutcome[]): DemoTally {
  const ok = outcomes.filter((o) => o.error === null);
  return {
    evaluated: ok.length,
    errored: outcomes.length - ok.length,
    fixed: summarizeEngine(ok, (o) => o.fixed),
    patientListener: summarizeEngine(ok, (o) => o.patientListener),
  };
}

export function formatEngineSummary(s: EngineSummary): string {
  const parts: string[] = [];
  if (s.earlyCount > 0) parts.push(`answered before the user finished in ${s.earlyCount} (mean ${s.meanEarlyMs} ms early)`);
  if (s.lateCount > 0) parts.push(`answered after the true end in ${s.lateCount} (mean delay ${s.meanDelayMs} ms)`);
  if (s.noAnswer > 0) parts.push(`no answer in ${s.noAnswer}`);
  return parts.length > 0 ? parts.join("; ") : "n/a";
}
