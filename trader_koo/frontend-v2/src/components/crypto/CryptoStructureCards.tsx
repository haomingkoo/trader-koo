import type {
  CryptoCorrelationPayload,
  CryptoMarketStructurePayload,
  CryptoStructurePayload,
} from "../../api/types";
import GlassCard from "../ui/GlassCard";
import {
  formatLevelContext,
  formatPct,
  formatPrice,
} from "./CryptoCardPrimitives";

export function StructureCard({
  structure,
}: {
  structure: CryptoStructurePayload | null | undefined;
}) {
  if (!structure) {
    return (
      <GlassCard label="Structure Engine">
        <div className="text-sm text-[var(--muted)]">
          Waiting for enough bars to map support and resistance.
        </div>
      </GlassCard>
    );
  }

  const context = structure.context;
  const regime = structure.hmm_regime;
  const regimeLabel = regime?.current_state?.replaceAll("_", " ") ?? "unavailable";
  const transitionRisk = regime?.transition_risk_pct ?? null;
  const directional = (structure as Record<string, unknown>).hmm_directional as { current_state?: string; days_in_current?: number; transition_risk_pct?: number } | null | undefined;
  const dirLabel = directional?.current_state?.replaceAll("_", " ") ?? null;
  const dirColors: Record<string, string> = { bullish: "var(--green)", bearish: "var(--red)", chop: "#a855f7" };

  return (
    <GlassCard label="Structure Engine">
      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-5">
        <div>
          <div className="label-xs">
            Level Context
          </div>
          <div className="mt-1 text-lg font-bold text-[var(--text)]">
            {formatLevelContext(context.level_context)}
          </div>
          <div className="mt-1 text-[11px] text-[var(--muted)]">
            Trend: {formatLevelContext(context.ma_trend)}
          </div>
        </div>
        <div>
          <div className="label-xs">
            Nearest Support
          </div>
          <div className="mt-1 text-lg font-bold tabular-nums text-[var(--blue)]">
            {context.support_level !== null ? `$${formatPrice(context.support_level)}` : "--"}
          </div>
          <div className="mt-1 text-[11px] text-[var(--muted)]">
            Distance {formatPct(context.pct_to_support)}
          </div>
        </div>
        <div>
          <div className="label-xs">
            Nearest Resistance
          </div>
          <div className="mt-1 text-lg font-bold tabular-nums text-[var(--red)]">
            {context.resistance_level !== null ? `$${formatPrice(context.resistance_level)}` : "--"}
          </div>
          <div className="mt-1 text-[11px] text-[var(--muted)]">
            Distance {formatPct(context.pct_to_resistance)}
          </div>
        </div>
        <div>
          <div className="label-xs">
            HMM Regime
          </div>
          <div className="mt-1 text-lg font-bold text-[var(--text)]">
            {formatLevelContext(regimeLabel)}
          </div>
          <div className="mt-1 text-[11px] text-[var(--muted)]">
            {regime?.days_in_current != null
              ? `${regime.days_in_current} bars${transitionRisk != null ? ` · ${transitionRisk.toFixed(1)}% shift risk` : ""}`
              : "Insufficient bars for stable regime fit"}
          </div>
        </div>
        <div>
          <div className="label-xs">
            Direction
          </div>
          <div className="mt-1 text-lg font-bold" style={{ color: dirLabel ? (dirColors[directional?.current_state ?? ""] ?? "var(--text)") : "var(--muted)" }}>
            {dirLabel ? formatLevelContext(dirLabel) : "Computing..."}
          </div>
          <div className="mt-1 text-[11px] text-[var(--muted)]">
            {directional?.days_in_current != null
              ? `${directional.days_in_current} bars${directional.transition_risk_pct != null ? ` · ${directional.transition_risk_pct.toFixed(1)}% shift risk` : ""}`
              : "Directional HMM"}
          </div>
        </div>
      </div>

      <div className="mt-4 grid gap-2 text-xs sm:grid-cols-2 xl:grid-cols-4">
        <div className="rounded-lg border border-[var(--line)] bg-[var(--bg)]/40 p-3">
          <div className="text-[9px] uppercase text-[var(--muted)]">Range Position</div>
          <div className="mt-1 font-semibold tabular-nums text-[var(--text)]">
            {context.range_position != null ? `${(context.range_position * 100).toFixed(0)}%` : "--"}
          </div>
        </div>
        <div className="rounded-lg border border-[var(--line)] bg-[var(--bg)]/40 p-3">
          <div className="text-[9px] uppercase text-[var(--muted)]">ATR %</div>
          <div className="mt-1 font-semibold tabular-nums text-[var(--text)]">
            {formatPct(context.atr_pct)}
          </div>
        </div>
        <div className="rounded-lg border border-[var(--line)] bg-[var(--bg)]/40 p-3">
          <div className="text-[9px] uppercase text-[var(--muted)]">Momentum 20</div>
          <div className="mt-1 font-semibold tabular-nums text-[var(--text)]">
            {formatPct(context.momentum_20)}
          </div>
        </div>
        <div className="rounded-lg border border-[var(--line)] bg-[var(--bg)]/40 p-3">
          <div className="text-[9px] uppercase text-[var(--muted)]">Realized Vol 20</div>
          <div className="mt-1 font-semibold tabular-nums text-[var(--text)]">
            {context.realized_vol_20 != null ? `${context.realized_vol_20.toFixed(2)}%` : "--"}
          </div>
        </div>
      </div>

      <div className="mt-4 rounded-lg border border-[var(--line)] bg-[var(--bg)]/40 p-3">
        <div className="mb-2 text-[9px] font-semibold uppercase tracking-wider text-[var(--muted)]">
          Auto Levels
        </div>
        <div className="space-y-2">
          {structure.levels.length > 0 ? structure.levels.map((level) => (
            <div
              key={`${level.type}-${level.level}`}
              className="flex items-center justify-between gap-3 text-xs"
            >
              <div>
                <span className={level.type === "support" ? "text-[var(--blue)]" : "text-[var(--red)]"}>
                  {level.type.toUpperCase()}
                </span>
                <span className="ml-2 text-[var(--muted)]">
                  {level.tier.toUpperCase()} · {level.source ?? "pivot_cluster"}
                </span>
              </div>
              <div className="font-semibold tabular-nums text-[var(--text)]">
                ${formatPrice(level.level)}
              </div>
            </div>
          )) : (
            <div className="text-sm text-[var(--muted)]">No nearby levels detected.</div>
          )}
        </div>
      </div>
    </GlassCard>
  );
}

const BENCHMARK_NAMES: Record<string, string> = {
  SPY: "S&P 500",
  GLD: "Gold",
  UUP: "Dollar",
  TLT: "Bonds",
  QQQ: "Nasdaq",
};

export function BtcSpyCorrelationCard({
  correlation,
}: {
  correlation: CryptoCorrelationPayload | null | undefined;
}) {
  const benchmark = correlation?.benchmark ?? "SPY";
  const benchmarkLabel = BENCHMARK_NAMES[benchmark] ?? benchmark;

  if (!correlation) {
    return (
      <GlassCard label={`BTC vs ${benchmarkLabel}`}>
        <div className="text-sm text-[var(--muted)]">
          Waiting for aligned daily closes.
        </div>
      </GlassCard>
    );
  }

  const windows = Object.entries(correlation.windows).sort(
    ([left], [right]) => Number.parseInt(left, 10) - Number.parseInt(right, 10),
  );

  // Build rebased mini sparkline from aligned history
  const history = correlation.aligned_history ?? [];

  return (
    <GlassCard label={`BTC vs ${benchmarkLabel} Cross-Asset`}>
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <div className="text-lg font-bold text-[var(--text)]">
            {formatLevelContext(correlation.relationship_label)}
          </div>
          <div className="mt-1 text-xs text-[var(--muted)]">
            {correlation.sample_size} aligned sessions
            {correlation.as_of ? ` · as of ${correlation.as_of}` : ""}
          </div>
        </div>
        <div className="grid grid-cols-2 gap-3 text-right text-xs">
          <div>
            <div className="text-[10px] uppercase tracking-wide text-[var(--muted)]">BTC</div>
            <div className="font-semibold tabular-nums text-[var(--text)]">
              ${correlation.latest.asset_close !== null ? formatPrice(correlation.latest.asset_close) : "--"}
            </div>
            <div
              className={
                correlation.latest.asset_change_1d_pct !== null &&
                correlation.latest.asset_change_1d_pct >= 0
                  ? "text-[var(--green)]"
                  : "text-[var(--red)]"
              }
            >
              {formatPct(correlation.latest.asset_change_1d_pct)}
            </div>
          </div>
          <div>
            <div className="text-[10px] uppercase tracking-wide text-[var(--muted)]">{benchmark}</div>
            <div className="font-semibold tabular-nums text-[var(--text)]">
              ${correlation.latest.benchmark_close !== null ? formatPrice(correlation.latest.benchmark_close) : "--"}
            </div>
            <div
              className={
                correlation.latest.benchmark_change_1d_pct !== null &&
                correlation.latest.benchmark_change_1d_pct >= 0
                  ? "text-[var(--green)]"
                  : "text-[var(--red)]"
              }
            >
              {formatPct(correlation.latest.benchmark_change_1d_pct)}
            </div>
          </div>
        </div>
      </div>

      <div className="mt-4 grid gap-3 md:grid-cols-3">
        {windows.map(([label, window]) => (
          <div
            key={label}
            className="rounded-lg border border-[var(--line)] bg-[var(--bg)]/45 p-3 text-xs"
          >
            <div className="text-[10px] font-semibold uppercase tracking-[0.16em] text-[var(--muted)]">
              {label}
            </div>
            <div className="mt-2 grid grid-cols-2 gap-2">
              <div>
                <div className="text-[9px] uppercase text-[var(--muted)]">Corr</div>
                <div className="font-semibold tabular-nums text-[var(--text)]">
                  {window.correlation !== null ? window.correlation.toFixed(2) : "--"}
                </div>
              </div>
              <div>
                <div className="text-[9px] uppercase text-[var(--muted)]">Beta</div>
                <div className="font-semibold tabular-nums text-[var(--text)]">
                  {window.beta !== null ? window.beta.toFixed(2) : "--"}
                </div>
              </div>
              <div>
                <div className="text-[9px] uppercase text-[var(--muted)]">BTC Ret</div>
                <div className="font-semibold tabular-nums text-[var(--text)]">
                  {formatPct(window.asset_return_pct)}
                </div>
              </div>
              <div>
                <div className="text-[9px] uppercase text-[var(--muted)]">Spread</div>
                <div className="font-semibold tabular-nums text-[var(--text)]">
                  {formatPct(window.relative_performance_pct)}
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Rebased price overlay sparkline */}
      {history.length >= 3 && (
        <div className="mt-3">
          <div className="text-[9px] uppercase tracking-wide text-[var(--muted)] mb-1">Rebased Performance (last {history.length} sessions)</div>
          <svg viewBox={`0 0 ${history.length * 10} 60`} className="w-full h-[60px]" preserveAspectRatio="none">
            {/* BTC line (green) */}
            <polyline
              fill="none"
              stroke="#38d39f"
              strokeWidth="1.5"
              points={history.map((h, i) => `${i * 10},${60 - ((h.asset_rebased - 90) / 20) * 60}`).join(" ")}
            />
            {/* Benchmark line (amber) */}
            <polyline
              fill="none"
              stroke="#f59e0b"
              strokeWidth="1.5"
              points={history.map((h, i) => `${i * 10},${60 - ((h.benchmark_rebased - 90) / 20) * 60}`).join(" ")}
            />
          </svg>
          <div className="flex justify-between text-[9px] text-[var(--muted)] mt-0.5">
            <span><span style={{color:"#38d39f"}}>--</span> BTC</span>
            <span><span style={{color:"#f59e0b"}}>--</span> {benchmarkLabel}</span>
          </div>
        </div>
      )}
    </GlassCard>
  );
}

export function CryptoBreadthCard({
  market,
}: {
  market: CryptoMarketStructurePayload | null | undefined;
}) {
  if (!market) {
    return (
      <GlassCard label="Crypto Breadth">
        <div className="text-sm text-[var(--muted)]">
          Waiting for enough 1h bars across tracked crypto pairs.
        </div>
      </GlassCard>
    );
  }

  const overview = market.overview;

  return (
    <GlassCard label="Crypto Breadth (1h)">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <div className="text-lg font-bold text-[var(--text)]">
            {formatLevelContext(overview.market_posture)}
          </div>
          <div className="mt-1 text-xs text-[var(--muted)]">
            Volatility regime: {formatLevelContext(overview.volatility_regime)}
          </div>
        </div>
        <div className="flex flex-wrap gap-2 text-[10px] uppercase tracking-[0.16em]">
          {market.leaders.slice(0, 2).map((row) => (
            <span
              key={`leader-${row.symbol}`}
              className="rounded-full border border-[var(--line)] bg-[rgba(56,211,159,0.1)] px-2 py-1 text-[var(--green)]"
            >
              Lead {row.symbol.replace("-USD", "")}
            </span>
          ))}
          {market.laggards.slice(0, 1).map((row) => (
            <span
              key={`laggard-${row.symbol}`}
              className="rounded-full border border-[var(--line)] bg-[rgba(255,107,107,0.1)] px-2 py-1 text-[var(--red)]"
            >
              Lag {row.symbol.replace("-USD", "")}
            </span>
          ))}
        </div>
      </div>

      <div className="mt-4 grid gap-2 text-xs sm:grid-cols-2 xl:grid-cols-4">
        <div className="rounded-lg border border-[var(--line)] bg-[var(--bg)]/40 p-3">
          <div className="text-[9px] uppercase text-[var(--muted)]">Bullish Trend Breadth</div>
          <div className="mt-1 font-semibold tabular-nums text-[var(--text)]">
            {overview.bullish_trend_count}/{overview.tracked_symbols}
          </div>
        </div>
        <div className="rounded-lg border border-[var(--line)] bg-[var(--bg)]/40 p-3">
          <div className="text-[9px] uppercase text-[var(--muted)]">Avg 24h Change</div>
          <div className="mt-1 font-semibold tabular-nums text-[var(--text)]">
            {formatPct(overview.avg_change_pct_24h)}
          </div>
        </div>
        <div className="rounded-lg border border-[var(--line)] bg-[var(--bg)]/40 p-3">
          <div className="text-[9px] uppercase text-[var(--muted)]">Avg Momentum 20</div>
          <div className="mt-1 font-semibold tabular-nums text-[var(--text)]">
            {formatPct(overview.avg_momentum_20)}
          </div>
        </div>
        <div className="rounded-lg border border-[var(--line)] bg-[var(--bg)]/40 p-3">
          <div className="text-[9px] uppercase text-[var(--muted)]">Support / Resistance</div>
          <div className="mt-1 font-semibold tabular-nums text-[var(--text)]">
            {overview.at_support_count} / {overview.at_resistance_count}
          </div>
        </div>
      </div>

      <div className="mt-4 space-y-2">
        {market.symbols.map((row) => (
          <div
            key={row.symbol}
            className="grid grid-cols-[1.1fr_0.8fr_0.9fr_0.9fr] gap-3 rounded-lg border border-[var(--line)] bg-[var(--bg)]/35 p-3 text-xs"
          >
            <div>
              <div className="font-semibold text-[var(--text)]">{row.symbol.replace("-USD", "")}</div>
              <div className="text-[var(--muted)]">{formatLevelContext(row.level_context)}</div>
            </div>
            <div className="tabular-nums">
              <div className="text-[9px] uppercase text-[var(--muted)]">24h</div>
              <div className="font-semibold text-[var(--text)]">{formatPct(row.change_pct_24h)}</div>
            </div>
            <div className="tabular-nums">
              <div className="text-[9px] uppercase text-[var(--muted)]">Trend</div>
              <div className="font-semibold text-[var(--text)]">{formatLevelContext(row.ma_trend)}</div>
            </div>
            <div className="tabular-nums">
              <div className="text-[9px] uppercase text-[var(--muted)]">ATR %</div>
              <div className="font-semibold text-[var(--text)]">{formatPct(row.atr_pct)}</div>
            </div>
          </div>
        ))}
      </div>
    </GlassCard>
  );
}
