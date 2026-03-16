import { Routes, Route } from "react-router-dom";
import DashboardLayout from "./routes/DashboardLayout";
import GuidePage from "./pages/GuidePage";
import ReportPage from "./pages/ReportPage";
import VixPage from "./pages/VixPage";
import EarningsPage from "./pages/EarningsPage";
import ChartPage from "./pages/ChartPage";
import OpportunitiesPage from "./pages/OpportunitiesPage";
import PaperTradePage from "./pages/PaperTradePage";
import NotFoundPage from "./pages/NotFoundPage";

export default function App() {
  return (
    <Routes>
      <Route element={<DashboardLayout />}>
        <Route index element={<GuidePage />} />
        <Route path="report" element={<ReportPage />} />
        <Route path="vix" element={<VixPage />} />
        <Route path="earnings" element={<EarningsPage />} />
        <Route path="chart" element={<ChartPage />} />
        <Route path="opportunities" element={<OpportunitiesPage />} />
        <Route path="paper-trades" element={<PaperTradePage />} />
        <Route path="*" element={<NotFoundPage />} />
      </Route>
    </Routes>
  );
}
