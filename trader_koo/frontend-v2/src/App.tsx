import { lazy, Suspense } from "react";
import { Routes, Route } from "react-router-dom";
import DashboardLayout from "./routes/DashboardLayout";
import Spinner from "./components/ui/Spinner";

/* Route-level code splitting — each page is loaded on demand */
const GuidePage = lazy(() => import("./pages/GuidePage"));
const ReportPage = lazy(() => import("./pages/ReportPage"));
const VixPage = lazy(() => import("./pages/VixPage"));
const EarningsPage = lazy(() => import("./pages/EarningsPage"));
const ChartPage = lazy(() => import("./pages/ChartPage"));
const OpportunitiesPage = lazy(() => import("./pages/OpportunitiesPage"));
const PaperTradePage = lazy(() => import("./pages/PaperTradePage"));
const CryptoPage = lazy(() => import("./pages/CryptoPage"));
const NotFoundPage = lazy(() => import("./pages/NotFoundPage"));

function PageFallback() {
  return <Spinner className="mt-12" />;
}

export default function App() {
  return (
    <Suspense fallback={<PageFallback />}>
      <Routes>
        <Route element={<DashboardLayout />}>
          <Route
            index
            element={
              <Suspense fallback={<PageFallback />}>
                <GuidePage />
              </Suspense>
            }
          />
          <Route
            path="report"
            element={
              <Suspense fallback={<PageFallback />}>
                <ReportPage />
              </Suspense>
            }
          />
          <Route
            path="vix"
            element={
              <Suspense fallback={<PageFallback />}>
                <VixPage />
              </Suspense>
            }
          />
          <Route
            path="earnings"
            element={
              <Suspense fallback={<PageFallback />}>
                <EarningsPage />
              </Suspense>
            }
          />
          <Route
            path="chart"
            element={
              <Suspense fallback={<PageFallback />}>
                <ChartPage />
              </Suspense>
            }
          />
          <Route
            path="opportunities"
            element={
              <Suspense fallback={<PageFallback />}>
                <OpportunitiesPage />
              </Suspense>
            }
          />
          <Route
            path="paper-trades"
            element={
              <Suspense fallback={<PageFallback />}>
                <PaperTradePage />
              </Suspense>
            }
          />
          <Route
            path="crypto"
            element={
              <Suspense fallback={<PageFallback />}>
                <CryptoPage />
              </Suspense>
            }
          />
          <Route
            path="*"
            element={
              <Suspense fallback={<PageFallback />}>
                <NotFoundPage />
              </Suspense>
            }
          />
        </Route>
      </Routes>
    </Suspense>
  );
}
