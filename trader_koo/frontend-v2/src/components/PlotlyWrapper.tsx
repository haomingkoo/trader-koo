import { lazy, Suspense } from "react";
import Spinner from "./ui/Spinner";

type PlotComponentModule = {
  default?: unknown;
};

function isRenderablePlotComponent(
  value: unknown,
): value is React.ComponentType<Record<string, unknown>> {
  return (
    typeof value === "function" ||
    (typeof value === "object" &&
      value !== null &&
      "$$typeof" in value)
  );
}

const Plot = lazy(async () => {
  const mod = (await import("react-plotly.js")) as PlotComponentModule;
  const nestedDefault = (mod.default as PlotComponentModule | undefined)?.default;
  const resolved = nestedDefault ?? mod.default;

  if (!isRenderablePlotComponent(resolved)) {
    throw new Error("Unable to resolve react-plotly.js component export");
  }

  return { default: resolved };
});

interface PlotlyWrapperProps {
  data: Array<Record<string, unknown>>;
  layout?: Record<string, unknown>;
  config?: Record<string, unknown>;
  className?: string;
  style?: React.CSSProperties;
  useResizeHandler?: boolean;
  onRelayout?: (event: Record<string, unknown>) => void;
}

/**
 * Consistent wrapper around react-plotly.js that handles lazy loading
 * and provides a Suspense fallback spinner.
 */
export default function PlotlyWrapper({
  data,
  layout,
  config,
  className,
  style,
  useResizeHandler = true,
  onRelayout,
}: PlotlyWrapperProps) {
  return (
    <Suspense
      fallback={
        <div className="flex items-center justify-center py-12">
          <Spinner size="lg" />
        </div>
      }
    >
      <Plot
        data={data}
        layout={layout}
        config={config}
        className={className}
        style={style}
        useResizeHandler={useResizeHandler}
        onRelayout={onRelayout}
      />
    </Suspense>
  );
}
