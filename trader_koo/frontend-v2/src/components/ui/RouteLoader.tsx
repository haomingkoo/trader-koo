import Spinner from "./Spinner";

interface RouteLoaderProps {
  title?: string;
  detail?: string;
}

export default function RouteLoader({
  title = "Loading dashboard view",
  detail = "Preparing charts, signals, and report context.",
}: RouteLoaderProps) {
  return (
    <div className="rounded-2xl border border-[var(--line)] bg-[var(--panel)]/85 px-6 py-10 backdrop-blur-sm">
      <div className="mx-auto flex max-w-xl flex-col items-center gap-4 text-center">
        <div className="flex h-16 w-16 items-center justify-center rounded-2xl border border-[var(--line)] bg-[var(--bg)]/70">
          <Spinner size="lg" />
        </div>
        <div className="space-y-1">
          <p className="text-[11px] font-semibold uppercase tracking-[0.3em] text-[var(--accent)]">
            trader_koo
          </p>
          <h2 className="text-lg font-semibold text-[var(--text)]">{title}</h2>
          <p className="text-sm text-[var(--muted)]">{detail}</p>
        </div>
      </div>
    </div>
  );
}
