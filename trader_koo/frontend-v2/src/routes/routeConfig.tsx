import { lazy } from "react";
import type { ComponentType, LazyExoticComponent } from "react";
import {
  BarChart3,
  Bell,
  Bitcoin,
  BookOpen,
  Calendar,
  DollarSign,
  FileText,
  Fish,
  Layers,
  Search,
  Thermometer,
  TrendingUp,
  Wallet,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";

type PageModule = { default: ComponentType };
type PageLoader = () => Promise<PageModule>;

interface AppRouteDefinition {
  path?: string;
  index?: boolean;
  navPath?: string;
  label?: string;
  Icon?: LucideIcon;
  loaderTitle: string;
  loaderDetail: string;
  load: PageLoader;
}

export interface ResolvedAppRoute extends AppRouteDefinition {
  Component: LazyExoticComponent<ComponentType>;
}

export interface NavRoute {
  to: string;
  label: string;
  Icon: LucideIcon;
}

export const appRoutes: AppRouteDefinition[] = [
  {
    index: true,
    navPath: "/",
    label: "Guide",
    Icon: BookOpen,
    loaderTitle: "Loading guide",
    loaderDetail: "Getting the dashboard tour and navigation shortcuts ready.",
    load: () => import("../pages/GuidePage"),
  },
  {
    path: "report",
    navPath: "/report",
    label: "Report",
    Icon: FileText,
    loaderTitle: "Loading daily report",
    loaderDetail: "Refreshing the latest setups, risk filters, and macro context.",
    load: () => import("../pages/ReportPage"),
  },
  {
    path: "chart",
    navPath: "/chart",
    label: "Chart",
    Icon: TrendingUp,
    loaderTitle: "Loading chart workspace",
    loaderDetail: "Preparing price history, overlays, commentary, and pattern context.",
    load: () => import("../pages/ChartPage"),
  },
  {
    path: "alerts",
    navPath: "/alerts",
    label: "Alerts",
    Icon: Bell,
    loaderTitle: "Loading alerts",
    loaderDetail: "Fetching recent price alerts, market spikes, and crypto moves.",
    load: () => import("../pages/AlertsPage"),
  },
  {
    path: "paper-trades",
    navPath: "/paper-trades",
    label: "Paper Trades",
    Icon: Wallet,
    loaderTitle: "Loading paper trades",
    loaderDetail: "Reconstructing performance, equity curve, and trade history.",
    load: () => import("../pages/PaperTradePage"),
  },
  {
    path: "vix",
    navPath: "/vix",
    label: "VIX Analysis",
    Icon: Thermometer,
    loaderTitle: "Loading volatility view",
    loaderDetail: "Building the VIX regime picture and volatility metrics.",
    load: () => import("../pages/VixPage"),
  },
  {
    path: "earnings",
    navPath: "/earnings",
    label: "Calendar",
    Icon: Calendar,
    loaderTitle: "Loading earnings calendar",
    loaderDetail: "Sorting the upcoming catalyst window into sessions and lanes.",
    load: () => import("../pages/EarningsPage"),
  },
  {
    path: "opportunities",
    navPath: "/opportunities",
    label: "Opportunities",
    Icon: Search,
    loaderTitle: "Loading opportunities",
    loaderDetail: "Screening valuation signals and setup quality across the universe.",
    load: () => import("../pages/OpportunitiesPage"),
  },
  {
    path: "options",
    navPath: "/options",
    label: "Options",
    Icon: DollarSign,
    loaderTitle: "Loading options premium",
    loaderDetail: "Aggregating the latest option-chain premium proxy snapshots.",
    load: () => import("../pages/OptionsPage"),
  },
  {
    path: "crypto",
    navPath: "/crypto",
    label: "Crypto",
    Icon: Bitcoin,
    loaderTitle: "Loading crypto dashboard",
    loaderDetail: "Streaming price cards, indicators, and intraday bar history.",
    load: () => import("../pages/CryptoPage"),
  },
  {
    path: "hyperliquid",
    navPath: "/hyperliquid",
    label: "Hyperliquid",
    Icon: Fish,
    loaderTitle: "Loading Hyperliquid tracker",
    loaderDetail: "Fetching whale positions and counter-trade signals.",
    load: () => import("../pages/HyperliquidPage"),
  },
  {
    path: "markets",
    navPath: "/markets",
    label: "Pred Markets",
    Icon: BarChart3,
    loaderTitle: "Loading prediction markets",
    loaderDetail: "Fetching live odds from Polymarket.",
    load: () => import("../pages/PolymarketPage"),
  },
  {
    path: "methodology",
    navPath: "/methodology",
    label: "Methodology",
    Icon: Layers,
    loaderTitle: "Loading methodology",
    loaderDetail: "Preparing the trading pipeline walkthrough.",
    load: () => import("../pages/MethodologyPage"),
  },
];

export const notFoundRoute: AppRouteDefinition = {
  path: "*",
  loaderTitle: "Loading page",
  loaderDetail: "Checking the requested route.",
  load: () => import("../pages/NotFoundPage"),
};

export const resolvedAppRoutes: ResolvedAppRoute[] = appRoutes.map((route) => ({
  ...route,
  Component: lazy(route.load),
}));

export const resolvedNotFoundRoute: ResolvedAppRoute = {
  ...notFoundRoute,
  Component: lazy(notFoundRoute.load),
};

export const navRoutes: NavRoute[] = appRoutes
  .filter(
    (route): route is AppRouteDefinition & Required<Pick<AppRouteDefinition, "navPath" | "label" | "Icon">> =>
      Boolean(route.navPath && route.label && route.Icon),
  )
  .map((route) => ({
    to: route.navPath,
    label: route.label,
    Icon: route.Icon,
  }));

export const routePreloaders: Record<string, () => Promise<unknown>> = Object.fromEntries(
  appRoutes
    .filter((route): route is AppRouteDefinition & { navPath: string } => Boolean(route.navPath))
    .map((route) => [route.navPath, route.load]),
);
