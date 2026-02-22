# Roadmap

## Phase 1: Data Foundation
- [ ] DuckDB schema (`prices_daily`, `fundamentals_finviz`, `signals_levels`, `signals_patterns`)
- [ ] Incremental OHLCV ingestion
- [ ] Fundamentals snapshot ingestion

## Phase 2: Signals
- [ ] Support/resistance + gap extraction job
- [ ] Pattern detection job (rule-based first)
- [ ] Optional CV confirmation pipeline

## Phase 3: Product
- [ ] FastAPI endpoints for ticker dashboard
- [ ] Dark-mode frontend dashboard
- [ ] Deploy locally first, then cloud host
