/**
 * Read CSS custom properties at call time so Plotly layouts
 * reflect the current theme (dark or light).
 *
 * Plotly can't consume CSS variables directly, so we read
 * computed values from `document.documentElement` and return
 * plain color strings.
 */
export function getPlotlyColors(): {
  font: string;
  grid: string;
  bg: string;
} {
  const style = getComputedStyle(document.documentElement);
  return {
    font: style.getPropertyValue("--muted").trim() || "#9ca3af",
    grid: style.getPropertyValue("--line").trim() || "rgba(255,255,255,0.06)",
    bg: "transparent",
  };
}
