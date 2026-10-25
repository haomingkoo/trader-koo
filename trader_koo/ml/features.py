"""Context symbols that are not trade candidates (imported by research code whose hashes pin this path)."""

# Symbols ingested for market regime/context should not be default ML trade
# candidates. Keep them available when explicitly requested for charts or
# diagnostics, but exclude them from full-universe model training/scoring.
ML_CONTEXT_TICKERS = {
    "SPY",
    "QQQ",
    "DIA",
    "IWM",
    "SVIX",
    "GLD",
    "USO",
    "UUP",
    "SLV",
    "^VIX",
    "^GSPC",
    "^DJI",
    "^TNX",
    "^VIX3M",
    "^VIX6M",
}
