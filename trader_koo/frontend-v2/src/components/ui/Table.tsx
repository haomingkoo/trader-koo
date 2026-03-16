import { useState, useMemo, useCallback } from "react";

interface ColumnDef<T> {
  key: keyof T & string;
  label: string;
  render?: (value: T[keyof T], row: T) => React.ReactNode;
}

interface TableProps<T extends Record<string, unknown>> {
  columns: ColumnDef<T>[];
  data: T[];
  onRowClick?: (row: T) => void;
  sortable?: boolean;
  className?: string;
}

export default function Table<T extends Record<string, unknown>>({
  columns,
  data,
  onRowClick,
  sortable = false,
  className = "",
}: TableProps<T>) {
  const [sortCol, setSortCol] = useState<string | null>(null);
  const [sortAsc, setSortAsc] = useState(true);

  const handleSort = useCallback(
    (key: string) => {
      if (!sortable) return;
      if (sortCol === key) {
        setSortAsc((prev) => !prev);
      } else {
        setSortCol(key);
        setSortAsc(true);
      }
    },
    [sortable, sortCol],
  );

  const sorted = useMemo(() => {
    if (!sortCol) return data;
    const dir = sortAsc ? 1 : -1;
    return [...data].sort((a, b) => {
      const av = a[sortCol];
      const bv = b[sortCol];
      const an = Number(av);
      const bn = Number(bv);
      if (Number.isFinite(an) && Number.isFinite(bn)) return (an - bn) * dir;
      return String(av ?? "").localeCompare(String(bv ?? "")) * dir;
    });
  }, [data, sortCol, sortAsc]);

  if (data.length === 0) {
    return (
      <div className="rounded-xl border border-[var(--line)] bg-[var(--panel)] p-6 text-center text-sm text-[var(--muted)]">
        No data available
      </div>
    );
  }

  return (
    <div className={`overflow-auto rounded-xl border border-[var(--line)] bg-[var(--panel)] ${className}`}>
      <table className="w-full border-collapse text-left text-sm">
        <thead>
          <tr className="border-b border-[var(--line)]">
            {columns.map((col) => (
              <th
                key={col.key}
                className={`px-3 py-2 text-xs font-semibold uppercase tracking-wider text-[var(--muted)] ${sortable ? "cursor-pointer select-none hover:text-[var(--text)]" : ""}`}
                onClick={() => handleSort(col.key)}
              >
                {col.label}
                {sortable && sortCol === col.key && (
                  <span className="ml-1">{sortAsc ? "\u25B2" : "\u25BC"}</span>
                )}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sorted.map((row, idx) => (
            <tr
              key={idx}
              className={`border-b border-[var(--line)] last:border-b-0 ${onRowClick ? "cursor-pointer hover:bg-[var(--panel-hover)]" : ""}`}
              onClick={() => onRowClick?.(row)}
            >
              {columns.map((col) => (
                <td key={col.key} className="px-3 py-2 text-[var(--text)]">
                  {col.render
                    ? col.render(row[col.key], row)
                    : String(row[col.key] ?? "\u2014")}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
