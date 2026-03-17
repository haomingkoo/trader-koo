import { Suspense, lazy } from "react";
import { Routes, Route } from "react-router-dom";
import DashboardLayout from "./routes/DashboardLayout";
import Spinner from "./components/ui/Spinner";

// Direct imports — no lazy loading until #306 is resolved
import GuidePage from "./pages/GuidePage";
import ReportPage from "./pages/ReportPage";
import VixPage from "./pages/VixPage";
import EarningsPage from "./pages/EarningsPage";
import OpportunitiesPage from "./pages/OpportunitiesPage";
import PaperTradePage from "./pages/PaperTradePage";
import NotFoundPage from "./pages/NotFoundPage";

// Keep Plotly-heavy pages lazy to avoid massive bundle
const ChartPage = lazy(() => import("./pages/ChartPage"));
const CryptoPage = lazy(() => import("./pages/CryptoPage"));

function PageFallback() {
  return <Spinner className="mt-12" />;
}

export default function App() {
  return (
    <Routes>
      <Route element={<DashboardLayout />}>
        <Route index element={<GuidePage />} />
        <Route path="report" element={<ReportPage />} />
        <Route path="vix" element={<VixPage />} />
        <Route path="earnings" element={<EarningsPage />} />
        <Route
          path="chart"
          element={
            <Suspense fallback={<PageFallback />}>
              <ChartPage />
            </Suspense>
          }
        />
        <Route path="opportunities" element={<OpportunitiesPage />} />
        <Route path="paper-trades" element={<PaperTradePage />} />
        <Route
          path="crypto"
          element={
            <Suspense fallback={<PageFallback />}>
              <CryptoPage />
            </Suspense>
          }
        />
        <Route path="*" element={<NotFoundPage />} />
      </Route>
    </Routes>
  );
}
