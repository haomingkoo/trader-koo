import { lazy, Suspense } from "react";
import { Routes, Route } from "react-router-dom";
import RouteLoader from "./components/ui/RouteLoader";
import DashboardLayout from "./routes/DashboardLayout";

const GuidePage = lazy(() => import("./pages/GuidePage"));
const ReportPage = lazy(() => import("./pages/ReportPage"));
const VixPage = lazy(() => import("./pages/VixPage"));
const EarningsPage = lazy(() => import("./pages/EarningsPage"));
const ChartPage = lazy(() => import("./pages/ChartPage"));
const OpportunitiesPage = lazy(() => import("./pages/OpportunitiesPage"));
const PaperTradePage = lazy(() => import("./pages/PaperTradePage"));
const CryptoPage = lazy(() => import("./pages/CryptoPage"));
const PolymarketPage = lazy(() => import("./pages/PolymarketPage"));
const MethodologyPage = lazy(() => import("./pages/MethodologyPage"));
const AlertsPage = lazy(() => import("./pages/AlertsPage"));
const NotFoundPage = lazy(() => import("./pages/NotFoundPage"));

function routeElement(
  element: React.ReactNode,
  title: string,
  detail: string,
) {
  return (
    <Suspense fallback={<RouteLoader title={title} detail={detail} />}>
      {element}
    </Suspense>
  );
}

export default function App() {
  return (
    <Routes>
      <Route element={<DashboardLayout />}>
        <Route
          index
          element={routeElement(
            <GuidePage />,
            "Loading guide",
            "Getting the dashboard tour and navigation shortcuts ready.",
          )}
        />
        <Route
          path="report"
          element={routeElement(
            <ReportPage />,
            "Loading daily report",
            "Refreshing the latest setups, risk filters, and macro context.",
          )}
        />
        <Route
          path="vix"
          element={routeElement(
            <VixPage />,
            "Loading volatility view",
            "Building the VIX regime picture and volatility metrics.",
          )}
        />
        <Route
          path="earnings"
          element={routeElement(
            <EarningsPage />,
            "Loading earnings calendar",
            "Sorting the upcoming catalyst window into sessions and lanes.",
          )}
        />
        <Route
          path="chart"
          element={routeElement(
            <ChartPage />,
            "Loading chart workspace",
            "Preparing price history, overlays, commentary, and pattern context.",
          )}
        />
        <Route
          path="opportunities"
          element={routeElement(
            <OpportunitiesPage />,
            "Loading opportunities",
            "Screening valuation signals and setup quality across the universe.",
          )}
        />
        <Route
          path="paper-trades"
          element={routeElement(
            <PaperTradePage />,
            "Loading paper trades",
            "Reconstructing performance, equity curve, and trade history.",
          )}
        />
        <Route
          path="markets"
          element={routeElement(
            <PolymarketPage />,
            "Loading prediction markets",
            "Fetching live odds from Polymarket.",
          )}
        />
        <Route
          path="crypto"
          element={routeElement(
            <CryptoPage />,
            "Loading crypto dashboard",
            "Streaming price cards, indicators, and intraday bar history.",
          )}
        />
        <Route
          path="methodology"
          element={routeElement(
            <MethodologyPage />,
            "Loading methodology",
            "Preparing the trading pipeline walkthrough.",
          )}
        />
        <Route
          path="alerts"
          element={routeElement(
            <AlertsPage />,
            "Loading alerts",
            "Fetching recent price alerts, market spikes, and crypto moves.",
          )}
        />
        <Route
          path="*"
          element={routeElement(
            <NotFoundPage />,
            "Loading page",
            "Checking the requested route.",
          )}
        />
      </Route>
    </Routes>
  );
}
