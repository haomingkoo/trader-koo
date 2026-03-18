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
  cryptoMarketStructure: CryptoMarketStructurePayload | undefined;
  indicators: CryptoIndicators | null;
}

export default function CryptoAnalyticsPanels({
  structure,
  btcSpyCorrelation,
  cryptoMarketStructure,
  indicators,
}: CryptoAnalyticsPanelsProps) {
  return (
    <>
      <StructureCard structure={structure} />

      <div className="grid gap-4 xl:grid-cols-2">
        <BtcSpyCorrelationCard correlation={btcSpyCorrelation} />
        <CryptoBreadthCard market={cryptoMarketStructure} />
      </div>

      {indicators && (
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <RsiGauge value={indicators.rsi_14} />
          <MacdCard macd={indicators.macd} />
          <BollingerCard bollinger={indicators.bollinger} />
          <VwapSmaCard
            vwap={indicators.vwap}
            sma20={indicators.sma_20}
            sma50={indicators.sma_50}
          />
        </div>
      )}
    </>
  );
}
