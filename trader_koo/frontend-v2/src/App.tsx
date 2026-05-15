import { Suspense } from "react";
import { Routes, Route } from "react-router-dom";
import RouteLoader from "./components/ui/RouteLoader";
import DashboardLayout from "./routes/DashboardLayout";
import {
  resolvedAppRoutes,
  resolvedNotFoundRoute,
} from "./routes/routeConfig";

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
  const NotFoundComponent = resolvedNotFoundRoute.Component;

  return (
    <Routes>
      <Route element={<DashboardLayout />}>
        {resolvedAppRoutes.map(({ Component, index, path, loaderTitle, loaderDetail }) => {
          const element = routeElement(<Component />, loaderTitle, loaderDetail);
          return index ? (
            <Route key="index" index element={element} />
          ) : (
            <Route key={path} path={path} element={element} />
          );
        })}
        <Route
          path={resolvedNotFoundRoute.path}
          element={routeElement(
            <NotFoundComponent />,
            resolvedNotFoundRoute.loaderTitle,
            resolvedNotFoundRoute.loaderDetail,
          )}
        />
      </Route>
    </Routes>
  );
}
