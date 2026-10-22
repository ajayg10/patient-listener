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
  user: string;
  persona: string;
  text_preview: string;
  true_end_t_ms: number;
  true_end_time?: number;
  silence_ms: number;
  category: string;
  demo?: boolean;
}

/** Replay SSE event payload emitted by POST /replay/{scenario} */
export interface ReplayMetadataEvent {
  event: "metadata";
  engine: "system";
  provenance: {
    backend: string;
    model_id: string;
    keys_count: number;
    note: string;
    fixed_timeout_ms: number;
    decision_latency_ms: number;
  };
  t_ms: number;
}

export interface ReplayAudioEvent {
  event: "speechstarted" | "speechdelta" | "speechended";
  engine: "audio";
  text?: string;
  t_ms: number;
}

export interface ReplayDecisionRequestedEvent {
  event: "decision_requested";
  engine: "patient_listener";
  checkpoint_ms: number;
  t_ms: number;
}

export interface ReplayDecisionCompletedEvent {
  event: "decision_completed";
  engine: "patient_listener";
  checkpoint_ms: number;
  choice: string;
  p_done: number;
  action: string;
  latency_ms: number;
  t_ms: number;
}

export interface ReplayDecisionDroppedEvent {
  event: "decision_dropped";
  engine: "patient_listener";
  reason: string;
  checkpoint_ms?: number;
  t_ms: number;
}

export interface ReplayRespondEvent {
  event: "respond";
  engine: "fixed" | "patient_listener";
  reason: string;
  silence_ms?: number;
  t_ms: number;
}

export interface ReplayFinalEvent {
  event: "final";
  engine: "fixed" | "patient_listener";
  interrupted: boolean;
  response_delay_ms: number;
  t_ms: number;
}

export interface ReplayErrorEvent {
  event: "error";
  engine?: string;
  error: string;
  message: string;
  checkpoint_ms?: number;
  state_key?: string;
  state_text?: string;
  t_ms?: number;
}

export type ReplayEvent =
  | ReplayMetadataEvent
  | ReplayAudioEvent
  | ReplayDecisionRequestedEvent
  | ReplayDecisionCompletedEvent
  | ReplayDecisionDroppedEvent
  | ReplayRespondEvent
  | ReplayFinalEvent
  | ReplayErrorEvent;

export interface HealthResponse {
  status: string;
  cache_provenance?: {
    backend: string;
    keys_count?: number;
    total_keys?: number;
    model_id?: string;
    is_mock?: boolean;
    note?: string;
    decision_latency_ms?: number;
    latency_source?: string;
    hardware?: string;
  };
  bedrock_status?: {
    status: string;
    model_id: string;
    region: string;
    mock_mode: boolean;
    error?: string;
  };
}

