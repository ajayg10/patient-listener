/**
 * Typed conversational turn and audio event schemas for Patient Listener.
 * Strictly mirrors backend/app/events.py.
 */

export interface Decision {
  choice: "RESPOND" | "WAIT" | string;
  probabilities: Record<string, number>;
  confidence: number;
  latency_ms: number;
  backend: "mock" | "local" | "remote" | string;
  round_trip_ms?: number | null;
}

/** Emitted when Voice Activity Detection (VAD) detects speech onset. */
export interface SpeechStarted {
  type: "SpeechStarted";
  timestamp_ms: number;
}

/** Incremental transcript fragment from Automatic Speech Recognition (ASR). */
export interface SpeechDelta {
  type: "SpeechDelta";
  text: string;
  timestamp_ms: number;
}

/** Emitted when VAD detects speech offset / silence onset. */
export interface SpeechEnded {
  type: "SpeechEnded";
  timestamp_ms: number;
}

/** Periodic probe emitted at configured pause checkpoints (e.g. 250, 500, 800 ms). */
export interface SilenceProbe {
  type: "SilenceProbe";
  silence_ms: number;
  timestamp_ms: number;
}

/** Emitted when an asynchronous Decider query is dispatched. */
export interface DecisionRequested {
  type: "DecisionRequested";
  checkpoint_ms: number;
  generation: number;
  timestamp_ms: number;
}

/** Emitted when an asynchronous Decider query completes successfully. */
export interface DecisionCompleted {
  type: "DecisionCompleted";
  decision: Decision;
  generation: number;
  checkpoint_ms: number;
  timestamp_ms: number;
}

/** Emitted when an in-flight decision is discarded (stale generation or interrupted by speech). */
export interface DecisionDropped {
  type: "DecisionDropped";
  generation: number;
  reason: string;
  timestamp_ms: number;
}

/** Emitted when the policy determines the conversational turn is complete and yields to agent reply. */
export interface TurnYielded {
  type: "TurnYielded";
  action: "RESPOND" | "WAIT" | string;
  reason: string;
  timestamp_ms: number;
}

/** Emitted when silence exceeds the hard safety maximum wait cap. */
export interface MaxHoldExceeded {
  type: "MaxHoldExceeded";
  max_hold_ms: number;
  timestamp_ms: number;
}

/** Union of all turn event schemas from backend/app/events.py. */
export type TurnEvent =
  | SpeechStarted
  | SpeechDelta
  | SpeechEnded
  | SilenceProbe
  | DecisionRequested
  | DecisionCompleted
  | DecisionDropped
  | TurnYielded
  | MaxHoldExceeded;

/** Server-to-client WebSocket message payloads */
export interface ServerDecisionMessage {
  type: "decision";
  probs: Record<string, number>;
  p_done: number;
  latency_ms: number;
  action: string;
  checkpoint: number;
  timestamp_ms: number;
}

export interface ServerTurnYieldedMessage {
  type: "turn_yielded";
  action: string;
  reason: string;
  timestamp_ms: number;
}

export interface ServerDecisionRequestedMessage {
  type: "decision_requested";
  checkpoint_ms: number;
  generation: number;
  timestamp_ms: number;
}

export interface ServerDecisionDroppedMessage {
  type: "decision_dropped";
  reason: string;
  generation: number;
  timestamp_ms: number;
}

export interface ServerMaxHoldExceededMessage {
  type: "max_hold_exceeded";
  max_hold_ms: number;
  timestamp_ms: number;
}

export interface ServerReplyChunkMessage {
  type: "reply_chunk";
  chunk: string;
  reply_source: "bedrock" | "mock" | string;
  timestamp_ms: number;
}

export interface ServerReplyDoneMessage {
  type: "reply_done";
  full_reply: string;
  reply_source: "bedrock" | "mock" | string;
  timestamp_ms: number;
}

export interface ServerErrorMessage {
  type: "error";
  message: string;
  error?: string;
  t_ms?: number;
}

export type ServerMessage =
  | ServerDecisionMessage
  | ServerTurnYieldedMessage
  | ServerDecisionRequestedMessage
  | ServerDecisionDroppedMessage
  | ServerMaxHoldExceededMessage
  | ServerReplyChunkMessage
  | ServerReplyDoneMessage
  | ServerErrorMessage;

/** Scenario item from GET /scenarios */
export interface ScenarioItem {
  id: string;
  user_id: string;
  persona_id: string;
  archetype: string;
  text_preview: string;
  true_end_time_ms: number;
  category: string;
  silence_ms: number;
  is_turn_complete: boolean;
}
