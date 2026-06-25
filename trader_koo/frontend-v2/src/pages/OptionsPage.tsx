import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { RefreshCw } from "lucide-react";
import { useOptionsPremium } from "../api/hooks";
import { OPTIONS_PREMIUM_CONFIG } from "../api/optionsConfig";
import type { OptionsPremiumSort, OptionsSmartView } from "../api/optionsConfig";
import type { OptionsPremiumRow } from "../api/types";
import Badge from "../components/ui/Badge";
import Spinner from "../components/ui/Spinner";
import Table from "../components/ui/Table";

type BadgeVariant = "green" | "amber" | "red" | "muted";

const formatMoney = (value: number | null | undefined): string => {
  if (typeof value !== "number" || !Number.isFinite(value)) return "-";
  const sign = value < 0 ? "-" : "";
  const abs = Math.abs(value);
  if (abs >= 1_000_000_000) return `${sign}$${(abs / 1_000_000_000).toFixed(2)}B`;
  if (abs >= 1_000_000) return `${sign}$${(abs / 1_000_000).toFixed(2)}M`;
  if (abs >= 1_000) return `${sign}$${(abs / 1_000).toFixed(1)}K`;
  return `${sign}$${abs.toFixed(0)}`;
};

const formatNumber = (value: number | null | undefined, digits = 0): string => {
  if (typeof value !== "number" || !Number.isFinite(value)) return "-";
  return value.toLocaleString(undefined, {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  });
};

const formatTimestamp = (ts: string | null | undefined): string => {
  if (!ts) return "-";
  try {
    return new Date(ts).toLocaleString();
  } catch {
    return ts;
  }
};

const premiumVariant = (
  value: string | null | undefined,
): BadgeVariant => {
  const key = String(value ?? "unknown") as keyof typeof OPTIONS_PREMIUM_CONFIG.premiumBiasVariants;
  return OPTIONS_PREMIUM_CONFIG.premiumBiasVariants[key] ?? OPTIONS_PREMIUM_CONFIG.premiumBiasVariants.unknown;
};

const premiumLabel = (value: string | null | undefined): string => {
  const key = String(value ?? "unknown") as keyof typeof OPTIONS_PREMIUM_CONFIG.premiumBiasLabels;
  return OPTIONS_PREMIUM_CONFIG.premiumBiasLabels[key] ?? OPTIONS_PREMIUM_CONFIG.premiumBiasLabels.unknown;
};

const signedMoneyClass = (value: number | null | undefined): string => {
  if (typeof value !== "number") return "";
  if (value > 0) return "text-[var(--green)]";
  if (value < 0) return "text-[var(--red)]";
  return "text-[var(--muted)]";
};

const scoreClass = (value: number | null | undefined): string => {
  if (typeof value !== "number") return "text-[var(--muted)]";
  if (value >= OPTIONS_PREMIUM_CONFIG.scoreBands.strong) return "text-[var(--green)]";
  if (value >= OPTIONS_PREMIUM_CONFIG.scoreBands.moderate) return "text-[var(--amber)]";
  return "text-[var(--muted)]";
};

const smartSignalLabel = (value: string | null | undefined): string => {
  const key = String(value ?? "") as keyof typeof OPTIONS_PREMIUM_CONFIG.smartSignals;
  return OPTIONS_PREMIUM_CONFIG.smartSignals[key] ?? OPTIONS_PREMIUM_CONFIG.defaultSignalLabel;
};

const tagLabel = (value: string): string => {
  const key = value as keyof typeof OPTIONS_PREMIUM_CONFIG.tagLabels;
  return OPTIONS_PREMIUM_CONFIG.tagLabels[key] ?? value.replaceAll("_", " ");
};

const tagVariant = (value: string): BadgeVariant => {
  const key = value as keyof typeof OPTIONS_PREMIUM_CONFIG.tagVariants;
  return OPTIONS_PREMIUM_CONFIG.tagVariants[key] ?? OPTIONS_PREMIUM_CONFIG.defaultTagVariant;
};

const includesText = (values: readonly string[], value: string | null | undefined): boolean =>
  values.includes(String(value ?? ""));

const positivePremiumFloor = (): number => OPTIONS_PREMIUM_CONFIG.positivePremiumFloor;

const matchesSmartView = (row: OptionsPremiumRow, view: OptionsSmartView): boolean => {
  const rules = OPTIONS_PREMIUM_CONFIG.smartViewRules;
  const tags = new Set(row.smart_tags ?? []);
  if (view === "all") return true;
  if (view === "best") {
    return includesText(rules.best.signals, row.smart_signal);
  }
  if (view === "value") {
    return (
      rules.value.requiredTags.every((tag) => tags.has(tag)) &&
      rules.value.excludedTags.every((tag) => !tags.has(tag))
    );
  }
  if (view === "calls") {
    return (
      includesText(rules.calls.premiumBiases, row.premium_bias) &&
      (row.net_volume_premium ?? positivePremiumFloor()) > positivePremiumFloor()
    );
  }
  if (view === "hedge") {
    return (
      includesText(rules.hedge.premiumBiases, row.premium_bias) ||
      rules.hedge.optionalTags.some((tag) => tags.has(tag))
    );
  }
  if (view === "hot") {
    return rules.hot.requiredTags.every((tag) => tags.has(tag));
  }
  return true;
};

const optionsColumns = [
  {
    key: "ticker" as const,
    label: "Ticker",
    render: (value: unknown) => {
      const ticker = String(value ?? "");
      if (!ticker) return "-";
      return (
        <Link
          to={`/chart?t=${ticker}`}
          className="font-mono font-bold text-[var(--accent)] transition-colors hover:text-[var(--blue)]"
        >
          {ticker}
        </Link>
      );
    },
  },
  {
    key: "smart_score" as const,
    label: "Edge",
    render: (value: unknown, row: OptionsPremiumRow) => {
      const n = typeof value === "number" ? value : null;
      return (
        <div className="flex flex-col gap-1">
          <span className={`font-mono text-sm font-semibold ${scoreClass(n)}`}>
            {n == null ? "-" : n.toFixed(0)}
          </span>
          <span className="text-[10px] uppercase text-[var(--muted)]">
            {smartSignalLabel(row.smart_signal)}
          </span>
        </div>
      );
    },
  },
  {
    key: "premium_bias" as const,
    label: "Bias",
    render: (value: unknown, row: OptionsPremiumRow) => (
      <div className="flex flex-col gap-1">
        <Badge variant={premiumVariant(String(value ?? ""))}>
          {premiumLabel(String(value ?? ""))}
        </Badge>
        <span className="text-[10px] uppercase text-[var(--muted)]">
          {row.primary_premium_source === "volume" ? "Volume" : "Open interest"}
        </span>
      </div>
    ),
  },
  {
    key: "smart_tags" as const,
    label: "Tags",
    render: (value: unknown) => {
      const tags = Array.isArray(value) ? value.slice(0, 3).map(String) : [];
      if (tags.length === 0) return "-";
      return (
        <div className="flex max-w-[170px] flex-wrap gap-1">
          {tags.map((tag) => (
            <Badge key={tag} variant={tagVariant(tag)}>
              {tagLabel(tag)}
            </Badge>
          ))}
        </div>
      );
    },
  },
  {
    key: "net_volume_premium" as const,
    label: "Net Vol $",
    render: (value: unknown) => {
      const n = typeof value === "number" ? value : null;
      return <span className={`font-medium ${signedMoneyClass(n)}`}>{formatMoney(n)}</span>;
    },
  },
  {
    key: "call_volume_premium" as const,
    label: "Call Vol $",
    render: (value: unknown) => formatMoney(typeof value === "number" ? value : null),
  },
  {
    key: "put_volume_premium" as const,
    label: "Put Vol $",
    render: (value: unknown) => formatMoney(typeof value === "number" ? value : null),
  },
  {
    key: "net_oi_premium" as const,
    label: "Net OI $",
    render: (value: unknown) => {
      const n = typeof value === "number" ? value : null;
      return <span className={`font-medium ${signedMoneyClass(n)}`}>{formatMoney(n)}</span>;
    },
  },
  {
    key: "put_call_oi_ratio" as const,
    label: "P/C OI",
    render: (value: unknown) => {
      const n = typeof value === "number" ? value : null;
      return n == null ? "-" : n.toFixed(2);
    },
  },
  {
    key: "avg_iv_pct" as const,
    label: "Avg IV",
    render: (value: unknown) => {
      const n = typeof value === "number" ? value : null;
      return n == null ? "-" : `${n.toFixed(1)}%`;
    },
  },
  {
    key: "contracts" as const,
    label: "Contracts",
    render: (value: unknown) => formatNumber(typeof value === "number" ? value : null),
  },
  {
    key: "historical_snapshots" as const,
    label: "Snaps",
    render: (value: unknown) => formatNumber(typeof value === "number" ? value : null),
  },
];

export default function OptionsPage() {
  useEffect(() => {
    document.title = "Options Premium - Trader Koo";
  }, []);

  const [sortBy, setSortBy] = useState<OptionsPremiumSort>(
    OPTIONS_PREMIUM_CONFIG.defaultSort,
  );
  const [smartView, setSmartView] = useState<OptionsSmartView>(
    OPTIONS_PREMIUM_CONFIG.defaultSmartView,
  );
  const { data, isLoading, error, isFetching, refetch } = useOptionsPremium({
    limit: OPTIONS_PREMIUM_CONFIG.pageLimit,
    sort_by: sortBy,
  });

  const rows = useMemo(() => data?.rows ?? [], [data?.rows]);
  const filteredRows = useMemo(
    () => rows.filter((row) => matchesSmartView(row, smartView)),
    [rows, smartView],
  );
  const filteredTotals = useMemo(
    () => ({
      net_volume_premium: filteredRows.reduce(
        (sum, row) => sum + (row.net_volume_premium ?? positivePremiumFloor()),
        positivePremiumFloor(),
      ),
      net_oi_premium: filteredRows.reduce(
        (sum, row) => sum + (row.net_oi_premium ?? positivePremiumFloor()),
        positivePremiumFloor(),
      ),
    }),
    [filteredRows],
  );
  const latestSnapshot = data?.latest_snapshot_ts ?? rows[0]?.snapshot_ts ?? null;
  const summary = useMemo(
    () => [
      {
        label: "Net volume premium",
        value: formatMoney(filteredTotals.net_volume_premium),
        className: signedMoneyClass(filteredTotals.net_volume_premium),
      },
      {
        label: "Net OI premium",
        value: formatMoney(filteredTotals.net_oi_premium),
        className: signedMoneyClass(filteredTotals.net_oi_premium),
      },
      {
        label: "Names",
        value: formatNumber(filteredRows.length),
        className: "text-[var(--text)]",
      },
      {
        label: "Snapshot",
        value: formatTimestamp(latestSnapshot),
        className: "text-[var(--text)]",
      },
    ],
    [filteredRows.length, filteredTotals.net_oi_premium, filteredTotals.net_volume_premium, latestSnapshot],
  );

  if (isLoading) return <Spinner className="mt-12" />;
  if (error) {
    return (
      <div className="mt-12 text-center text-sm text-[var(--red)]">
        Failed to load options premium: {String((error as Error)?.message ?? "Unknown error")}
      </div>
    );
  }

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h2 className="text-xl font-bold tracking-tight">
          Options Premium Proxy
        </h2>
        <div className="flex flex-wrap items-center gap-2">
          <div className="flex flex-wrap gap-1" role="tablist" aria-label="Smart options view">
            {(Object.keys(OPTIONS_PREMIUM_CONFIG.smartViews) as OptionsSmartView[]).map((view) => (
              <button
                key={view}
                onClick={() => setSmartView(view)}
                className={`rounded-md px-3 py-1 text-xs font-semibold transition-colors ${
                  smartView === view
                    ? "bg-[var(--green)] text-[var(--bg)]"
                    : "border border-[var(--line)] bg-[var(--panel)] text-[var(--muted)] hover:text-[var(--text)]"
                }`}
                role="tab"
                aria-selected={smartView === view}
              >
                {OPTIONS_PREMIUM_CONFIG.smartViews[view]}
              </button>
            ))}
          </div>
          <div className="flex gap-1">
            {(Object.keys(OPTIONS_PREMIUM_CONFIG.sortLabels) as OptionsPremiumSort[]).map((mode) => (
              <button
                key={mode}
                onClick={() => setSortBy(mode)}
                className={`rounded-md px-3 py-1 text-xs font-semibold transition-colors ${
                  sortBy === mode
                    ? "bg-[var(--blue)] text-white"
                    : "border border-[var(--line)] bg-[var(--panel)] text-[var(--muted)] hover:text-[var(--text)]"
                }`}
              >
                {OPTIONS_PREMIUM_CONFIG.sortLabels[mode]}
              </button>
            ))}
          </div>
          <button
            onClick={() => void refetch()}
            className="inline-flex items-center gap-1.5 rounded-md border border-[var(--line)] bg-[var(--panel)] px-3 py-1 text-xs font-semibold text-[var(--muted)] transition-colors hover:text-[var(--text)]"
            aria-label="Refresh options premium"
            title="Refresh options premium"
          >
            <RefreshCw size={14} className={isFetching ? "animate-spin" : ""} />
            Refresh
          </button>
        </div>
      </div>

      <div className="grid gap-2 sm:grid-cols-2 xl:grid-cols-4">
        {summary.map((item) => (
          <div
            key={item.label}
            className="rounded-xl border border-[var(--line)] bg-[var(--panel)] px-4 py-3"
          >
            <div className="label-xs">{item.label}</div>
            <div className={`mt-1 text-lg font-semibold ${item.className}`}>
              {item.value}
            </div>
          </div>
        ))}
      </div>

      <div className="rounded-xl border border-[var(--line)] bg-[var(--panel)] px-4 py-3 text-xs text-[var(--muted)]">
        {data?.premium_proxy_note ?? OPTIONS_PREMIUM_CONFIG.unavailableMessage}
      </div>

      {rows.length === 0 ? (
        <div className="rounded-xl border border-[var(--line)] bg-[var(--panel)] p-8 text-center text-sm text-[var(--muted)]">
          {data?.detail ?? OPTIONS_PREMIUM_CONFIG.emptyMessage}
        </div>
      ) : filteredRows.length === 0 ? (
        <div className="rounded-xl border border-[var(--line)] bg-[var(--panel)] p-8 text-center text-sm text-[var(--muted)]">
          {OPTIONS_PREMIUM_CONFIG.noSmartMatchMessage}
        </div>
      ) : (
        <Table columns={optionsColumns} data={filteredRows} sortable />
      )}
    </div>
  );
}
