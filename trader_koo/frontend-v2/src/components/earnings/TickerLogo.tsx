import { useState } from "react";

interface TickerLogoProps {
  ticker: string;
  size?: number;
}

/** Hash a ticker string to a deterministic hue (0-360). */
function tickerHue(ticker: string): number {
  let hash = 0;
  for (let i = 0; i < ticker.length; i++) {
    hash = ticker.charCodeAt(i) + ((hash << 5) - hash);
  }
  return Math.abs(hash) % 360;
}

/**
 * Displays a company logo loaded from /logos/{TICKER}.png.
 * Falls back to a colored circle with the first letter on error.
 */
export default function TickerLogo({ ticker, size = 24 }: TickerLogoProps) {
  const [failed, setFailed] = useState(false);

  if (failed || !ticker) {
    const hue = tickerHue(ticker || "?");
    return (
      <span
        className="inline-flex shrink-0 items-center justify-center rounded-full text-white"
        style={{
          width: size,
          height: size,
          fontSize: size * 0.45,
          fontWeight: 700,
          backgroundColor: `hsl(${hue}, 55%, 45%)`,
        }}
      >
        {(ticker || "?")[0].toUpperCase()}
      </span>
    );
  }

  return (
    <img
      src={`/logos/${ticker}.png`}
      alt={ticker}
      width={size}
      height={size}
      className="shrink-0 rounded-full object-cover"
      style={{ width: size, height: size }}
      loading="lazy"
      onError={() => setFailed(true)}
    />
  );
}
