import GlassCard from "../ui/GlassCard";

interface FundamentalsData {
  price: number | null;
  pe: number | null;
  peg: number | null;
  target_price: number | null;
  discount_pct: number | null;
}

interface ChartFundamentalsProps {
  currentPrice: number | null;
  fundamentals: FundamentalsData;
  putCallOiRatio: number | null;
  formatNumber: (value: number | null | undefined, decimals?: number) => string;
}

export default function ChartFundamentals({
  currentPrice,
  fundamentals,
  putCallOiRatio,
  formatNumber,
}: ChartFundamentalsProps) {
  return (
    <div className="grid gap-3 sm:grid-cols-3 lg:grid-cols-6">
      <GlassCard
        label="Price"
        value={currentPrice != null ? `$${currentPrice.toFixed(2)}` : "\u2014"}
      />
      <GlassCard
        label="P/E"
        value={formatNumber(fundamentals.pe, 1)}
      />
      <GlassCard
        label="PEG"
        value={formatNumber(fundamentals.peg, 2)}
      />
      <GlassCard
        label="Target"
        value={fundamentals.target_price != null ? `$${fundamentals.target_price.toFixed(2)}` : "\u2014"}
      />
      <GlassCard
        label="Discount %"
        value={
          fundamentals.discount_pct != null
            ? `${fundamentals.discount_pct > 0 ? "+" : ""}${fundamentals.discount_pct.toFixed(1)}%`
            : "\u2014"
        }
        className={
          fundamentals.discount_pct != null && fundamentals.discount_pct > 0
            ? "border-[rgba(56,211,159,0.3)]"
            : fundamentals.discount_pct != null && fundamentals.discount_pct < -10
              ? "border-[rgba(255,107,107,0.3)]"
              : ""
        }
      />
      <GlassCard
        label="Put/Call OI"
        value={putCallOiRatio != null ? putCallOiRatio.toFixed(3) : "\u2014"}
      />
    </div>
  );
}
