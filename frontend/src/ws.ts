/**
 * Robust WebSocket client with auto-reconnect and typed event dispatching.
 */

import type { ServerMessage, TurnEvent } from "./events";

export type ConnectionStatus = "disconnected" | "connecting" | "connected" | "reconnecting" | "error";

export interface WebSocketClientOptions {
  url?: string;
  userId?: string;
  ticker?: boolean;
  onStatusChange?: (status: ConnectionStatus) => void;
  onMessage?: (message: ServerMessage) => void;
  onError?: (error: Event) => void;
  maxReconnectAttempts?: number;
  initialReconnectDelayMs?: number;
  maxReconnectDelayMs?: number;
}

export class PatientListenerWSClient {
  private ws: WebSocket | null = null;
  private status: ConnectionStatus = "disconnected";
  private reconnectAttempts = 0;
  private reconnectTimer: number | null = null;
  private isIntentionallyClosed = false;

  private readonly options: WebSocketClientOptions;

  constructor(options: WebSocketClientOptions = {}) {
    this.options = {
      maxReconnectAttempts: 10,
      initialReconnectDelayMs: 500,
      maxReconnectDelayMs: 5000,
      userId: "default_user",
      ticker: true,
      ...options,
    };
  }

  public getStatus(): ConnectionStatus {
    return this.status;
  }

  private setStatus(newStatus: ConnectionStatus) {
    if (this.status !== newStatus) {
      this.status = newStatus;
      this.options.onStatusChange?.(newStatus);
    }
  }

  private getWebSocketUrl(): string {
    if (this.options.url) {
      return this.options.url;
    }
    const apiBase = (import.meta.env.VITE_API_URL as string) || "http://localhost:8000";
    const wsBase = apiBase.replace(/^http:\/\//, "ws://").replace(/^https:\/\//, "wss://");
    const user = encodeURIComponent(this.options.userId || "default_user");
    const ticker = this.options.ticker !== false ? "true" : "false";
    return `${wsBase}/ws/session?user=${user}&ticker=${ticker}`;
  }

  public connect(): void {
    if (this.ws && (this.ws.readyState === WebSocket.OPEN || this.ws.readyState === WebSocket.CONNECTING)) {
      return;
    }

    this.isIntentionallyClosed = false;
    this.setStatus(this.reconnectAttempts > 0 ? "reconnecting" : "connecting");

    const url = this.getWebSocketUrl();
    try {
      this.ws = new WebSocket(url);

      this.ws.onopen = () => {
        this.reconnectAttempts = 0;
        this.setStatus("connected");
      };

      this.ws.onmessage = (event: MessageEvent) => {
        try {
          const parsed = JSON.parse(event.data) as ServerMessage;
          this.options.onMessage?.(parsed);
        } catch (err) {
          console.error("Failed to parse incoming WebSocket message:", err, event.data);
        }
      };

      this.ws.onerror = (error: Event) => {
        this.options.onError?.(error);
      };

      this.ws.onclose = () => {
        this.ws = null;
        if (!this.isIntentionallyClosed) {
          this.setStatus("disconnected");
          this.scheduleReconnect();
        } else {
          this.setStatus("disconnected");
        }
      };
    } catch (err) {
      console.error("Error creating WebSocket:", err);
      this.setStatus("error");
      this.scheduleReconnect();
    }
  }

  private scheduleReconnect(): void {
    const maxAttempts = this.options.maxReconnectAttempts ?? 10;
    if (this.reconnectAttempts >= maxAttempts) {
      this.setStatus("error");
      return;
    }

    const baseDelay = this.options.initialReconnectDelayMs ?? 500;
    const maxDelay = this.options.maxReconnectDelayMs ?? 5000;
    // Exponential backoff with 20% random jitter
    const delay = Math.min(maxDelay, baseDelay * Math.pow(1.5, this.reconnectAttempts)) * (0.9 + Math.random() * 0.2);
    this.reconnectAttempts++;

    this.setStatus("reconnecting");
    if (this.reconnectTimer) {
      window.clearTimeout(this.reconnectTimer);
    }
    this.reconnectTimer = window.setTimeout(() => {
      this.connect();
    }, delay);
  }

  public send(event: TurnEvent | Record<string, unknown>): boolean {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(event));
      return true;
    }
    console.warn("Cannot send message: WebSocket is not open.");
    return false;
  }

  public disconnect(): void {
    this.isIntentionallyClosed = true;
    if (this.reconnectTimer) {
      window.clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.setStatus("disconnected");
  }
}
