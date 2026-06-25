declare module "react-plotly.js/factory" {
  import type { ComponentType } from "react";

  const createPlotlyComponent: (plotly: unknown) => ComponentType<Record<string, unknown>>;
  export default createPlotlyComponent;
}

declare module "plotly.js/dist/plotly-finance.min.js" {
  const Plotly: unknown;
  export default Plotly;
}
