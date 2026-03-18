import type { ReactNode } from "react";
import type { LevelRow } from "../../api/types";
import Card from "../ui/Card";
import Badge from "../ui/Badge";
import Table from "../ui/Table";

interface LevelsCardProps {
  levels: LevelRow[];
  formatNumber: (value: number | null | undefined, decimals?: number) => string;
}

export default function LevelsCard({
  levels,
  formatNumber,
}: LevelsCardProps) {
  if (levels.length === 0) {
    return (
      <Card label="Support / Resistance Levels">
        <p className="mt-1 text-xs text-[var(--muted)]">
          No levels data available.
        </p>
      </Card>
    );
  }

  const columns: Array<{
    key: keyof LevelRow & string;
    label: string;
    render?: (value: unknown) => ReactNode;
  }> = [
    {
      key: "type",
      label: "Type",
      render: (value: unknown) => {
        const levelType = String(value ?? "");
        return (
          <Badge variant={levelType === "support" ? "blue" : "red"}>
            {levelType.toUpperCase()}
          </Badge>
        );
      },
    },
    {
      key: "level",
      label: "Level",
      render: (value: unknown) => formatNumber(value as number | null, 2),
    },
    { key: "tier", label: "Tier" },
    { key: "touches", label: "Touches" },
    { key: "last_touch_date", label: "Last Touch" },
  ];

  return (
    <div>
      <h3 className="mb-2 text-sm font-semibold text-[var(--muted)]">
        Support / Resistance ({levels.length})
      </h3>
      <Table
        columns={columns}
        data={levels as unknown as Record<string, unknown>[]}
        sortable
      />
    </div>
  );
}
