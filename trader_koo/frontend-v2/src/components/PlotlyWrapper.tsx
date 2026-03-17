import { lazy, Suspense } from "react";
import Spinner from "./ui/Spinner";

const Plot = lazy(() => import("react-plotly.js"));

interface PlotlyWrapperProps {
  data: Array<Record<string, unknown>>;
  layout?: Record<string, unknown>;
  config?: Record<string, unknown>;
  className?: string;
  style?: React.CSSProperties;
  useResizeHandler?: boolean;
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
      />
    </Suspense>
  );
}
