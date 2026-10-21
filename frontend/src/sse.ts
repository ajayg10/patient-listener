/**
 * Pure SSE streaming reader consuming ReadableStream<Uint8Array> from POST /replay/{scenario}.
 */

import type { ReplayEvent } from "./events";

export async function parseSSEStream(
  stream: ReadableStream<Uint8Array>,
  onEvent: (event: ReplayEvent) => void,
  signal?: AbortSignal
): Promise<void> {
  const reader = stream.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  try {
    while (true) {
      if (signal?.aborted) {
        await reader.cancel();
        break;
      }

      const { done, value } = await reader.read();
      if (done) {
        break;
      }

      buffer += decoder.decode(value, { stream: true });

      // SSE frames are separated by double newlines (\n\n or \r\n\r\n)
      const parts = buffer.split(/\r?\n\r?\n/);
      // The last element is the remaining uncompleted fragment
      buffer = parts.pop() ?? "";

      for (const part of parts) {
        const trimmed = part.trim();
        if (!trimmed) continue;

        const lines = trimmed.split(/\r?\n/);
        for (const line of lines) {
          if (line.startsWith("data:")) {
            const jsonText = line.slice(5).trim();
            if (jsonText) {
              try {
                const parsed = JSON.parse(jsonText) as ReplayEvent;
                onEvent(parsed);
              } catch (err) {
                console.error("Failed to parse SSE JSON frame:", jsonText, err);
              }
            }
          }
        }
      }
    }

    // Flush any remaining buffer if it contains a complete event
    if (buffer.trim()) {
      const lines = buffer.trim().split(/\r?\n/);
      for (const line of lines) {
        if (line.startsWith("data:")) {
          const jsonText = line.slice(5).trim();
          if (jsonText) {
            try {
              const parsed = JSON.parse(jsonText) as ReplayEvent;
              onEvent(parsed);
            } catch (err) {
              console.error("Failed to parse final SSE buffer:", jsonText, err);
            }
          }
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}
