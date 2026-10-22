import test from "node:test";
import assert from "node:assert/strict";

// We test the logic of formatResultLine as a pure function
function formatResultLine(ctx) {
  if (ctx.streamError) {
    const shortMsg = ctx.streamError.length > 60 ? `${ctx.streamError.slice(0, 57)}...` : ctx.streamError;
    return `Patient Listener produced no answer (stream error: ${shortMsg})`;
  }

  const hasResponded = ctx.hasResponse ?? (ctx.respondedAtMs !== null && ctx.respondedAtMs !== undefined);
  if (!hasResponded) {
    return "Patient Listener produced no answer";
  }

  if (ctx.interrupted === true) {
    const atMs = Math.round(ctx.respondedAtMs ?? 0);
    return `Patient Listener also interrupted at ${atMs} ms`;
  }

  if (ctx.interrupted === false && ctx.responseDelayMs !== null && ctx.responseDelayMs !== undefined) {
    const delay = Math.round(ctx.responseDelayMs);
    return `Patient Listener waited and answered ${delay} ms after the true end`;
  }

  return "Patient Listener produced no answer";
}

test("Case 1: Patient Listener waited and answered after true end", () => {
  const result = formatResultLine({
    interrupted: false,
    responseDelayMs: 551.2,
    respondedAtMs: 4051.2,
    hasResponse: true,
  });
  assert.equal(result, "Patient Listener waited and answered 551 ms after the true end");
});

test("Case 2: Patient Listener also interrupted", () => {
  const result = formatResultLine({
    interrupted: true,
    respondedAtMs: 800,
    hasResponse: true,
  });
  assert.equal(result, "Patient Listener also interrupted at 800 ms");
});

test("Case 3a: Stream error occurred", () => {
  const result = formatResultLine({
    interrupted: false,
    streamError: "cache_miss: key not found in precomputed decisions",
  });
  assert.equal(
    result,
    "Patient Listener produced no answer (stream error: cache_miss: key not found in precomputed decisions)"
  );
});

test("Case 3b: No response event occurred", () => {
  const result = formatResultLine({
    interrupted: null,
    respondedAtMs: null,
    hasResponse: false,
  });
  assert.equal(result, "Patient Listener produced no answer");
  assert.ok(!result.includes("waited and answered"));
});

function formatDelayLine(respondedAtMs, trueEndMs) {
  if (respondedAtMs === null || respondedAtMs === undefined) {
    return "no answer";
  }

  if (trueEndMs !== null && trueEndMs !== undefined) {
    if (respondedAtMs < trueEndMs) {
      const diff = Math.round(trueEndMs - respondedAtMs);
      return `answered ${diff} ms before the user finished`;
    }
    const diff = Math.round(respondedAtMs - trueEndMs);
    return `answered ${diff} ms after the true end`;
  }

  return `${Math.round(respondedAtMs)} ms`;
}

test("formatDelayLine: answered before the user finished", () => {
  // e.g. true end at 4500ms, engine interrupted at 800ms
  const line = formatDelayLine(800, 4500);
  assert.equal(line, "answered 3700 ms before the user finished");
});

test("formatDelayLine: answered after the true end", () => {
  // e.g. true end at 3500ms, engine responded at 4051ms
  const line = formatDelayLine(4051, 3500);
  assert.equal(line, "answered 551 ms after the true end");
});

test("formatDelayLine: no response", () => {
  assert.equal(formatDelayLine(null, 3500), "no answer");
  assert.equal(formatDelayLine(undefined, 3500), "no answer");
});

