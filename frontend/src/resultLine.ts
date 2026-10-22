/**
 * Pure functions to derive the replay outcome verdict result line.
 * Derives exclusively from final events and error states.
 */

export interface FinalEventContext {
  interrupted?: boolean | null;
  responseDelayMs?: number | null;
  respondedAtMs?: number | null;
  hasResponse?: boolean;
  streamError?: string | null;
}

/**
 * Derives the result line strictly from final events.
 * 
 * Cases:
 * 1. "Patient Listener waited and answered X ms after the true end" (interrupted=false and response time exists)
 * 2. "Patient Listener also interrupted at X ms" (interrupted=true)
 * 3. "Patient Listener produced no answer (stream error: <short message>)" (error occurred or no response exists)
 * 
 * Never displays a success message in error or no-response cases.
 */
export function formatResultLine(ctx: FinalEventContext): string {
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
