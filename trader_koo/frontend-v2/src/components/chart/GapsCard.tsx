import type { ReactNode } from "react";
import type { GapRow } from "../../api/types";
import Card from "../ui/Card";
import Badge from "../ui/Badge";
import Table from "../ui/Table";

interface GapsCardProps {
  gaps: GapRow[];
  formatNumber: (value: number | null | undefined, decimals?: number) => string;
}

export default function GapsCard({
  gaps,
  formatNumber,
}: GapsCardProps) {
  if (gaps.length === 0) {
    return (
      <Card label="Gaps">
        <p className="mt-1 text-xs text-[var(--muted)]">
          No gap data available.
        </p>
      </Card>
    );
  }

  const columns: Array<{
    key: keyof GapRow & string;
    label: string;
    render?: (value: unknown) => ReactNode;
  }> = [
    { key: "date", label: "Date" },
    {
      key: "type",
      label: "Type",
      render: (value: unknown) => {
        const gapType = String(value ?? "");
        return (
          <Badge variant={gapType === "bull_gap" ? "green" : "blue"}>
            {gapType.replace(/_/g, " ").toUpperCase()}
          </Badge>
        );
      },
    },
    {
      key: "gap_low",
      label: "Low",
      render: (value: unknown) => formatNumber(value as number | null, 2),
    },
    {
      key: "gap_high",
      label: "High",
      render: (value: unknown) => formatNumber(value as number | null, 2),
    },
  ];

  return (
    <div>
      <h3 className="mb-2 text-sm font-semibold text-[var(--muted)]">
        Gaps ({gaps.length})
      </h3>
      <Table
        columns={columns}
        data={gaps as unknown as Record<string, unknown>[]}
        sortable
      />
    </div>
  );
}
