import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { Bell, TrendingUp, BarChart3, Bitcoin } from "lucide-react";
import { useAlerts } from "../api/hooks";
import Badge from "../components/ui/Badge";
import Spinner from "../components/ui/Spinner";
import type { AlertItem } from "../api/types";

type FilterTab = "all" | "price_alert" | "market_spike" | "crypto_spike";

interface FilterTabConfig {
  key: FilterTab;
  label: string;
}

const FILTER_TABS: FilterTabConfig[] = [
  { key: "all", label: "All" },
  { key: "price_alert", label: "Price Alerts" },
  { key: "market_spike", label: "Market Spikes" },
  { key: "crypto_spike", label: "Crypto" },
];

const TYPE_ICON: Record<string, typeof TrendingUp> = {
  price_alert: TrendingUp,
  market_spike: BarChart3,
  crypto_spike: Bitcoin,
};

const SEVERITY_VARIANT: Record<string, "red" | "amber" | "muted"> = {
  high: "red",
  medium: "amber",
  low: "muted",
};

function extractTicker(title: string): string | null {
  const match = title.match(/^([A-Z]{1,5})\s/);
  return match ? match[1] : null;
}

function AlertCard({ alert }: { alert: AlertItem }) {
  const Icon = TYPE_ICON[alert.type] ?? TrendingUp;
  const severityVariant = SEVERITY_VARIANT[alert.severity] ?? "muted";

  const borderColor =
    alert.severity === "high"
      ? "border-l-[var(--red)]"
      : alert.severity === "medium"
        ? "border-l-[var(--amber)]"
        : "border-l-[var(--line)]";

  // Auto-link price alerts to chart page
  const ticker = alert.type === "price_alert" ? extractTicker(alert.title) : null;
  const linkPath = alert.internal_path || (ticker ? `/chart?ticker=${ticker}` : null);
  // linkPath handles both internal_path and auto-generated chart links

  const content = (
    <div className="flex items-start gap-3">
      <div className="flex h-9 w-9 flex-shrink-0 items-center justify-center rounded-lg bg-[var(--panel-hover)]">
        <Icon size={16} className="text-[var(--accent)]" />
      </div>
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2">
          <h3 className="truncate text-sm font-semibold text-[var(--text)]">
            {alert.title}
          </h3>
          <Badge variant={severityVariant}>{alert.severity}</Badge>
        </div>
        <p className="mt-1 text-xs text-[var(--muted)]">{alert.message}</p>
        <div className="mt-1.5 flex items-center gap-2">
          <span className="text-[10px] text-[var(--muted)]">
            {alert.time_ago} &middot;{" "}
            {new Date(alert.timestamp).toLocaleString()}
          </span>
          {alert.external_url && (
            <a
              href={alert.external_url}
              target="_blank"
              rel="noopener noreferrer"
              onClick={(e) => e.stopPropagation()}
              className="text-[10px] font-medium text-[var(--accent)] hover:underline"
            >
              Polymarket &rarr;
            </a>
          )}
          {alert.internal_path && (
            <span className="text-[10px] font-medium text-[var(--accent)]">
              View &rarr;
            </span>
          )}
        </div>
      </div>
    </div>
  );

  if (linkPath) {
    return (
      <Link
        to={linkPath}
        className={`block rounded-lg border border-[var(--line)] border-l-4 ${borderColor} bg-[var(--panel)] p-4 transition-colors hover:bg-[var(--panel-hover)] cursor-pointer`}
      >
        {content}
      </Link>
    );
  }

  return (
    <div
      className={`rounded-lg border border-[var(--line)] border-l-4 ${borderColor} bg-[var(--panel)] p-4 transition-colors hover:bg-[var(--panel-hover)]`}
    >
      {content}
    </div>
  );
}

export default function AlertsPage() {
  useEffect(() => {
    document.title = "Alerts \u2014 Trader Koo";
  }, []);
  const [activeTab, setActiveTab] = useState<FilterTab>("all");
  const { data, isLoading, isError } = useAlerts(200);

  const allAlerts = data?.alerts ?? [];
  const filtered =
    activeTab === "all"
      ? allAlerts
      : allAlerts.filter((a) => a.type === activeTab);

  return (
    <div className="mx-auto max-w-3xl space-y-4">
      <div className="flex items-center gap-3">
        <div className="flex h-9 w-9 items-center justify-center rounded-lg border border-[var(--line)] bg-[var(--panel)]">
          <Bell size={16} className="text-[var(--accent)]" />
        </div>
        <div>
          <h1 className="text-lg font-bold text-[var(--text)]">Alerts</h1>
          <p className="text-xs text-[var(--muted)]">
            Price alerts, prediction market spikes, and crypto moves
          </p>
        </div>
      </div>

      {/* Filter tabs */}
      <div className="flex gap-1 rounded-lg border border-[var(--line)] bg-[var(--panel)] p-1">
        {FILTER_TABS.map((tab) => {
          const count =
            tab.key === "all"
              ? allAlerts.length
              : allAlerts.filter((a) => a.type === tab.key).length;
          return (
            <button
              key={tab.key}
              onClick={() => setActiveTab(tab.key)}
              className={`flex-1 rounded-md px-3 py-1.5 text-xs font-medium transition-colors ${
                activeTab === tab.key
                  ? "bg-[var(--panel-hover)] text-[var(--accent)]"
                  : "text-[var(--muted)] hover:text-[var(--text)]"
              }`}
            >
              {tab.label}
              {count > 0 && (
                <span className="ml-1 text-[10px] opacity-60">({count})</span>
              )}
            </button>
          );
        })}
      </div>

      {/* Content */}
      {isLoading ? (
        <div className="flex items-center justify-center py-16">
          <Spinner size="lg" />
        </div>
      ) : isError ? (
        <div className="rounded-lg border border-[var(--line)] bg-[var(--panel)] px-6 py-10 text-center">
          <p className="text-sm text-[var(--red)]">
            Failed to load alerts. Please try again.
          </p>
        </div>
      ) : filtered.length === 0 ? (
        <div className="rounded-lg border border-[var(--line)] bg-[var(--panel)] px-6 py-16 text-center">
          <Bell
            size={32}
            className="mx-auto mb-3 text-[var(--muted)]"
          />
          <p className="text-sm font-medium text-[var(--text)]">
            No alerts yet
          </p>
          <p className="mx-auto mt-1 max-w-xs text-xs text-[var(--muted)]">
            Alerts will appear here when price levels are approached or
            prediction market odds shift.
          </p>
        </div>
      ) : (
        <div className="space-y-2">
          {filtered.map((alert) => (
            <AlertCard key={alert.id} alert={alert} />
          ))}
        </div>
      )}
    </div>
  );
}
