import { Component } from "react";
import type { ErrorInfo, ReactNode } from "react";

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
  /** Change this value to auto-reset the boundary (e.g. on route change). */
  resetKey?: string;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

const CHUNK_RELOAD_KEY = "trader_koo_v2_chunk_reload_at";
const CHUNK_RELOAD_WINDOW_MS = 30_000;

function isChunkLoadError(error: Error | null): boolean {
  const message = String(error?.message ?? "");
  return (
    message.includes("Failed to fetch dynamically imported module") ||
    message.includes("ChunkLoadError") ||
    message.includes("Importing a module script failed")
  );
}

export default class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidUpdate(prevProps: Props): void {
    if (
      this.state.hasError &&
      prevProps.resetKey !== this.props.resetKey
    ) {
      this.setState({ hasError: false, error: null });
    }
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    if (isChunkLoadError(error) && typeof window !== "undefined") {
      try {
        const lastReloadAt = Number(window.sessionStorage.getItem(CHUNK_RELOAD_KEY) ?? "0");
        if (!Number.isFinite(lastReloadAt) || Date.now() - lastReloadAt > CHUNK_RELOAD_WINDOW_MS) {
          window.sessionStorage.setItem(CHUNK_RELOAD_KEY, String(Date.now()));
          window.location.reload();
          return;
        }
      } catch {
        // Ignore sessionStorage failures and fall through to the visible fallback.
      }
    }
    console.error("ErrorBoundary caught:", error, info);
  }

  handleRetry = (): void => {
    if (isChunkLoadError(this.state.error) && typeof window !== "undefined") {
      window.location.reload();
      return;
    }
    this.setState({ hasError: false, error: null });
  };

  render(): ReactNode {
    if (this.state.hasError) {
      if (this.props.fallback) return this.props.fallback;
      return (
        <div className="flex flex-col items-center justify-center gap-4 rounded-xl border border-[var(--line)] bg-[var(--panel)] p-8 text-center">
          <div className="text-lg font-semibold text-[var(--red)]">
            Something went wrong
          </div>
          <div className="max-w-md text-sm text-[var(--muted)]">
            {isChunkLoadError(this.state.error)
              ? "A fresh deploy landed while this page was open. Reload to pick up the latest app shell."
              : String(this.state.error?.message ?? "An unexpected error occurred.")}
          </div>
          <button
            onClick={this.handleRetry}
            className="rounded-lg border border-[var(--line)] bg-[var(--panel-hover)] px-4 py-2 text-sm font-medium text-[var(--text)] transition-colors hover:bg-[var(--accent)] hover:text-white"
          >
            {isChunkLoadError(this.state.error) ? "Reload app" : "Retry"}
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}
