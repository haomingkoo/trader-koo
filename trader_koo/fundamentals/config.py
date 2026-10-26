"""Constants for the SEC point-in-time fundamentals store."""
from __future__ import annotations

SEC_USER_AGENT = "trader-koo research haomingkoo@gmail.com"
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"

# SEC fair-access ceiling is 10 requests/second; stay at 4/s.
REQUESTS_PER_SECOND = 4.0
HTTP_TIMEOUT_SEC = 60
MAX_ATTEMPTS = 5
BACKOFF_BASE_SEC = 2.0
BACKOFF_CAP_SEC = 120.0
HTTP_OK = 200
HTTP_NOT_FOUND = 404
RETRY_STATUSES = frozenset({403, 429, 500, 502, 503, 504})

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
QUARTER_DAYS = (80, 100)
YTD_DAYS = (150, 290)
ANNUAL_DAYS = (350, 380)
# Prior-year comparison periods may drift by this many days (52/53-week years).
MATCH_TOLERANCE_DAYS = 15
DAYS_PER_YEAR = 365
