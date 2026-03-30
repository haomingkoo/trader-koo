import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import App from "./App";
import { CryptoWsProvider } from "./hooks/useCryptoWs";
import { EquityWsProvider } from "./hooks/useEquityWs";
import "./styles/globals.css";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { retry: 1, refetchOnWindowFocus: false },
  },
});

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <CryptoWsProvider>
          <EquityWsProvider>
            <App />
          </EquityWsProvider>
        </CryptoWsProvider>
      </BrowserRouter>
    </QueryClientProvider>
  </StrictMode>,
);
