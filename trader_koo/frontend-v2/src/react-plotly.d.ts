declare module "react-plotly.js" {
  import type { Component } from "react";

  interface PlotParams {
    data: Array<Record<string, unknown>>;
    layout?: Record<string, unknown>;
    config?: Record<string, unknown>;
    style?: React.CSSProperties;
    className?: string;
    useResizeHandler?: boolean;
    onInitialized?: (figure: Record<string, unknown>, graphDiv: HTMLElement) => void;
    onUpdate?: (figure: Record<string, unknown>, graphDiv: HTMLElement) => void;
    onPurge?: (figure: Record<string, unknown>, graphDiv: HTMLElement) => void;
    onError?: (err: Error) => void;
  }

  class Plot extends Component<PlotParams> {}
  export default Plot;
}

declare module "react-plotly.js/factory" {
  import type { ComponentType } from "react";

  const createPlotlyComponent: (plotly: unknown) => ComponentType<Record<string, unknown>>;
  export default createPlotlyComponent;
}

declare module "plotly.js/dist/plotly-finance.min.js" {
  const Plotly: unknown;
  export default Plotly;
}
