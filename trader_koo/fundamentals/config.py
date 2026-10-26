"""Constants for the SEC point-in-time fundamentals store.

Tunables come from config/research.toml (trader_koo/research/settings.py); the
HTTP status codes and the concept map are protocol and reference data.
"""
from __future__ import annotations

from trader_koo.research.settings import get as _settings

_F = _settings().fundamentals

SEC_USER_AGENT = _F.sec_user_agent
SEC_TICKERS_URL = _F.sec_tickers_url
SEC_FACTS_URL = _F.sec_facts_url
REQUESTS_PER_SECOND = _F.requests_per_second
HTTP_TIMEOUT_SEC = _F.http_timeout_sec
MAX_ATTEMPTS = _F.max_attempts
BACKOFF_BASE_SEC = _F.backoff_base_sec
BACKOFF_CAP_SEC = _F.backoff_cap_sec
RETRY_STATUSES = frozenset(_F.retry_statuses)
HTTP_OK = 200
HTTP_NOT_FOUND = 404

# Concept -> candidate (taxonomy, tag) pairs in priority order. A company that
# changed tags over time is covered because the accessor takes the candidate
# with the latest period_end.
CONCEPTS: dict[str, tuple[tuple[str, str], ...]] = {
    "revenue": (
        ("us-gaap", "RevenueFromContractWithCustomerExcludingAssessedTax"),
        ("us-gaap", "RevenueFromContractWithCustomerIncludingAssessedTax"),
        ("us-gaap", "Revenues"),
        ("us-gaap", "SalesRevenueNet"),
    ),
    "gross_profit": (("us-gaap", "GrossProfit"),),
    "operating_income": (("us-gaap", "OperatingIncomeLoss"),),
    "net_income": (("us-gaap", "NetIncomeLoss"),),
    "eps_diluted": (("us-gaap", "EarningsPerShareDiluted"),),
    "assets": (("us-gaap", "Assets"),),
    "liabilities": (("us-gaap", "Liabilities"),),
    "equity": (
        ("us-gaap", "StockholdersEquity"),
        ("us-gaap", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"),
    ),
    "cash": (("us-gaap", "CashAndCashEquivalentsAtCarryingValue"),),
    "long_term_debt": (
        ("us-gaap", "LongTermDebt"),
        ("us-gaap", "LongTermDebtNoncurrent"),
    ),
    "operating_cash_flow": (("us-gaap", "NetCashProvidedByUsedInOperatingActivities"),),
    "capex": (("us-gaap", "PaymentsToAcquirePropertyPlantAndEquipment"),),
    "shares_outstanding": (
        ("us-gaap", "CommonStockSharesOutstanding"),
        ("dei", "EntityCommonStockSharesOutstanding"),
    ),
    "dividends_paid": (
        ("us-gaap", "PaymentsOfDividendsCommonStock"),
        ("us-gaap", "PaymentsOfDividends"),
    ),
}
STORED_TAGS = frozenset(pair for pairs in CONCEPTS.values() for pair in pairs)

# Duration windows in days between period start and end.
QUARTER_DAYS = _F.quarter_days
YTD_DAYS = _F.ytd_days
ANNUAL_DAYS = _F.annual_days
# Prior-year comparison periods may drift by this many days (52/53-week years).
MATCH_TOLERANCE_DAYS = _F.match_tolerance_days
DAYS_PER_YEAR = _F.days_per_year
