import { lazy, memo, Suspense } from "react";
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

// CJS↔ESM interop: a dynamically-imported CommonJS module's callable can sit at
// the namespace itself, at `.default`, or double-wrapped at `.default.default`
// depending on the bundler. Unwrap `.default` until we find the function.
function resolveCallable(mod: unknown): ((arg: unknown) => unknown) | null {
  let cur = mod;
  for (let depth = 0; depth < 4; depth += 1) {
    if (typeof cur === "function") {
      return cur as (arg: unknown) => unknown;
    }
    if (cur && typeof cur === "object" && "default" in cur) {
      cur = (cur as { default: unknown }).default;
    } else {
      return null;
    }
  }
  return null;
}

const Plot = lazy(async () => {
  const [factoryModule, plotlyModule] = await Promise.all([
    import("react-plotly.js/factory"),
    import("plotly.js/dist/plotly-finance.min.js"),
  ]);
  const plotly = (plotlyModule as PlotComponentModule).default ?? plotlyModule;
  const createPlotlyComponent = resolveCallable(factoryModule);
  if (createPlotlyComponent === null) {
    throw new Error("react-plotly.js factory did not resolve to a function");
  }
  const resolved = createPlotlyComponent(plotly);

  if (!isRenderablePlotComponent(resolved)) {
    throw new Error("Unable to create Plotly component");
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
function PlotlyWrapper({
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

export default memo(PlotlyWrapper);
