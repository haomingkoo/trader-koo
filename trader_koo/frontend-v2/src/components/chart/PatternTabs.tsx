import { useState } from "react";
import type {
  CandlestickPatternRow,
  DashboardPayload,
  PatternRow,
} from "../../api/types";
import Table from "../ui/Table";

interface PatternTabsProps {
  payload: DashboardPayload;
  formatNumber: (value: number | null | undefined, decimals?: number) => string;
}

export default function PatternTabs({
  payload,
  formatNumber,
}: PatternTabsProps) {
  const [activeTab, setActiveTab] = useState<"rule" | "candlestick">("rule");

  const rulePatterns = payload.patterns ?? [];
  const candlestickPatterns = payload.candlestick_patterns ?? [];

  const tabs = [
    { key: "rule" as const, label: `Rule (${rulePatterns.length})` },
    {
      key: "candlestick" as const,
      label: `Candlestick (${candlestickPatterns.length})`,
    },
  ];

  const ruleColumns: Array<{
    key: keyof PatternRow & string;
    label: string;
    render?: (value: unknown) => React.ReactNode;
  }> = [
    { key: "pattern", label: "Pattern" },
    { key: "status", label: "Status" },
    {
      key: "confidence",
      label: "Confidence",
      render: (value: unknown) => formatNumber(value as number | null, 2),
    },
    { key: "start_date", label: "Start" },
    { key: "end_date", label: "End" },
  ];

  const candleColumns: Array<{
    key: keyof CandlestickPatternRow & string;
    label: string;
    render?: (value: unknown) => React.ReactNode;
  }> = [
    { key: "date", label: "Date" },
    { key: "pattern", label: "Pattern" },
    { key: "bias", label: "Bias" },
    {
      key: "confidence",
      label: "Confidence",
      render: (value: unknown) => formatNumber(value as number | null, 2),
    },
    { key: "explanation", label: "Explanation" },
  ];

  return (
    <div>
      <div className="mb-2 flex gap-1">
        {tabs.map((tab) => (
          <button
            key={tab.key}
            onClick={() => setActiveTab(tab.key)}
            className={`rounded-md px-3 py-1 text-xs font-semibold transition-colors ${
              activeTab === tab.key
                ? "bg-[var(--accent)] text-white"
                : "border border-[var(--line)] bg-[var(--panel)] text-[var(--muted)] hover:text-[var(--text)]"
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {activeTab === "rule" && (
        rulePatterns.length > 0 ? (
          <Table
            columns={ruleColumns}
            data={rulePatterns as unknown as Record<string, unknown>[]}
            sortable
          />
        ) : (
          <div className="rounded-xl border border-[var(--line)] bg-[var(--panel)] p-6 text-center text-sm text-[var(--muted)]">
            No rule patterns detected.
          </div>
        )
      )}
      {activeTab === "candlestick" && (
        candlestickPatterns.length > 0 ? (
          <Table
            columns={candleColumns}
            data={candlestickPatterns as unknown as Record<string, unknown>[]}
            sortable
          />
        ) : (
          <div className="rounded-xl border border-[var(--line)] bg-[var(--panel)] p-6 text-center text-sm text-[var(--muted)]">
            No candlestick patterns detected.
          </div>
        )
      )}
    </div>
  );
}
