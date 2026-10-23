// Run with: node --test src/demoTally.test.mjs  (Node 24 loads .ts directly)
import test from "node:test";
import assert from "node:assert/strict";
import { summarizeDemoSet, formatEngineSummary } from "./demoTally.ts";

const scenario = (id, fixed, pl, error = null) => ({
  scenarioId: id,
  trueEndMs: 5000,
  fixed,
  patientListener: pl,
  error,
});
const cut = (at) => ({ interrupted: true, respondedAtMs: at, responseDelayMs: 0 });
const waited = (at, delay) => ({ interrupted: false, respondedAtMs: at, responseDelayMs: delay });

test("cut-ins never exceed scenarios evaluated", () => {
  const t = summarizeDemoSet([
    scenario("a", cut(4000), waited(5550, 550)),
    scenario("b", cut(3000), waited(5550, 550)),
    scenario("c", cut(4500), waited(5551, 551)),
  ]);
  assert.equal(t.evaluated, 3);
  assert.equal(t.fixed.cutIns, 3);
  assert.ok(t.fixed.cutIns <= t.evaluated);
  assert.equal(t.patientListener.cutIns, 0);
});

test("early answers are reported as time early, not 0 ms delay", () => {
  const t = summarizeDemoSet([scenario("a", cut(4000), waited(5550, 550)), scenario("b", cut(3000), waited(5550, 550))]);
  assert.equal(t.fixed.meanEarlyMs, 1500);
  assert.equal(t.fixed.meanDelayMs, null);
  assert.equal(t.patientListener.meanDelayMs, 550);
  assert.match(formatEngineSummary(t.fixed), /before the user finished in 2 \(mean 1500 ms early\)/);
});

test("errored scenarios are excluded and counted separately", () => {
  const t = summarizeDemoSet([scenario("a", cut(4000), null, "cache miss"), scenario("b", cut(4000), waited(5550, 550))]);
  assert.equal(t.evaluated, 1);
  assert.equal(t.errored, 1);
  assert.equal(t.fixed.cutIns, 1);
});

test("missing response is no answer, not a delay", () => {
  const t = summarizeDemoSet([scenario("a", waited(5800, 800), { interrupted: false, respondedAtMs: null, responseDelayMs: null })]);
  assert.equal(t.patientListener.noAnswer, 1);
  assert.equal(formatEngineSummary(t.patientListener), "no answer in 1");
});
