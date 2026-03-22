import type {
  CryptoIndicators,
  CryptoMarketStructurePayload,
  CryptoStructurePayload,
  CryptoCorrelationPayload,
} from "../../api/types";
import {
  BollingerCard,
  BtcSpyCorrelationCard,
  CryptoBreadthCard,
  MacdCard,
  RsiGauge,
  StructureCard,
  VwapSmaCard,
} from "./CryptoInsightCards";

interface CryptoAnalyticsPanelsProps {
  structure: CryptoStructurePayload | undefined;
  btcSpyCorrelation: CryptoCorrelationPayload | undefined;
  btcGoldCorrelation?: CryptoCorrelationPayload | undefined;
  btcDxyCorrelation?: CryptoCorrelationPayload | undefined;
  cryptoMarketStructure: CryptoMarketStructurePayload | undefined;
  indicators: CryptoIndicators | null;
}

export default function CryptoAnalyticsPanels({
  structure,
  btcSpyCorrelation,
  btcGoldCorrelation,
  btcDxyCorrelation,
  cryptoMarketStructure,
  indicators,
}: CryptoAnalyticsPanelsProps) {
  return (
    <>
      {/* Indicators right below chart (RSI is now a chart subplot) */}
      {indicators && (
        <div className="grid gap-3 sm:grid-cols-3">
          <MacdCard macd={indicators.macd} />
          <BollingerCard bollinger={indicators.bollinger} />
          <VwapSmaCard
            vwap={indicators.vwap}
            sma20={indicators.sma_20}
            sma50={indicators.sma_50}
          />
        </div>
      )}

      <StructureCard structure={structure} />

      <div className="grid gap-4 xl:grid-cols-2">
        <BtcSpyCorrelationCard correlation={btcSpyCorrelation} />
        <CryptoBreadthCard market={cryptoMarketStructure} />
      </div>

      {/* Cross-asset correlations: Gold & Dollar */}
      {(btcGoldCorrelation || btcDxyCorrelation) && (
        <div className="grid gap-4 xl:grid-cols-2">
          {btcGoldCorrelation && <BtcSpyCorrelationCard correlation={btcGoldCorrelation} />}
          {btcDxyCorrelation && <BtcSpyCorrelationCard correlation={btcDxyCorrelation} />}
        </div>
      )}
    </>
  );
}
