from __future__ import annotations

import datetime as dt
from dataclasses import dataclass


@dataclass
class CryptoTick:
    symbol: str  # "BTC-USD", "ETH-USD"
    price: float
    volume_24h: float
    change_pct_24h: float
    timestamp: dt.datetime


@dataclass
class CryptoBar:
    symbol: str
    timestamp: dt.datetime
    interval: str  # "1m", "5m", "1h", "1d"
    open: float
    high: float
    low: float
    close: float
    volume: float
