import type { ReactNode } from "react";
import type { YoloAuditRow } from "../../api/types";
import Table from "../ui/Table";

interface YoloAuditSectionProps {
  yoloAudit: YoloAuditRow[];
  formatNumber: (value: number | null | undefined, decimals?: number) => string;
}

export default function YoloAuditSection({
  yoloAudit,
  formatNumber,
}: YoloAuditSectionProps) {
  if (yoloAudit.length === 0) {
    return (
      <div>
        <h3 className="mb-2 text-sm font-semibold text-[var(--muted)]">
          YOLO Audit
        </h3>
        <div className="rounded-xl border border-[var(--line)] bg-[var(--panel)] p-6 text-center text-sm text-[var(--muted)]">
          No YOLO audit data available.
        </div>
      </div>
    );
  }

  const columns: Array<{
    key: keyof YoloAuditRow & string;
    label: string;
    render?: (value: unknown, row: unknown) => ReactNode;
  }> = [
    { key: "timeframe", label: "TF" },
    { key: "pattern", label: "Pattern" },
    { key: "signal_role", label: "Role" },
    {
      key: "active_now",
      label: "Active",
      render: (value: unknown) => (value ? "Yes" : "No"),
    },
    { key: "yolo_recency", label: "Recency" },
    { key: "confirmation_trend", label: "Trend" },
    { key: "lifecycle_state", label: "Lifecycle" },
    {
      key: "age_days",
      label: "Age (d)",
      render: (value: unknown) => formatNumber(value as number | null, 0),
    },
    {
      key: "current_streak",
      label: "Streak",
      render: (value: unknown) => formatNumber(value as number | null, 0),
    },
    {
      key: "confidence",
      label: "Conf",
      render: (value: unknown) => formatNumber(value as number | null, 2),
    },
    { key: "first_seen_asof", label: "First Seen" },
    { key: "last_seen_asof", label: "Last Seen" },
  ];

  return (
    <div>
      <h3 className="mb-2 text-sm font-semibold text-[var(--muted)]">
        YOLO Audit ({yoloAudit.length} entries)
      </h3>
      <Table
        columns={columns}
        data={yoloAudit as unknown as Record<string, unknown>[]}
        sortable
      />
    </div>
  );
}
