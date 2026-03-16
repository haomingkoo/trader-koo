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
