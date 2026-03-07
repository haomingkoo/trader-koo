# Design Document: Enterprise Platform Upgrade

## Overview

### Purpose

This design document specifies the comprehensive upgrade of trader_koo from a personal swing trading analysis tool to an enterprise-grade platform. The upgrade encompasses security hardening, VIX methodology improvements, core enterprise features, and competitive capabilities to position trader_koo as a professional-grade trading intelligence platform.

### Scope

The design covers 57 requirements across four priority levels:

- **P0 (Critical)**: 17 requirements - security hardening, VIX methodology, repository management
- **P1 (High)**: 16 requirements - enterprise features, non-functional requirements
- **P2 (Medium)**: 8 requirements - competitive feature parity
- **P3 (Low)**: 10 requirements - advanced competitive features

This document provides detailed design for P0 and P1 requirements, with high-level architecture for P2/P3 features.

### Goals

1. Eliminate security vulnerabilities and implement defense-in-depth
2. Improve VIX analysis accuracy and reliability with multi-source redundancy
3. Enable multi-user enterprise deployment with RBAC, audit logging, and monitoring
4. Achieve feature parity with mid-tier competitors (TradingView, StockCharts)
5. Establish foundation for institutional adoption

### Non-Goals

- Real-time tick-by-tick data streaming (P3 optional)
- Mobile native applications (mobile API only)
- Proprietary trading algorithm development
- Direct broker integration for order execution
- Cryptocurrency or forex analysis

### Success Criteria

- Zero secret exposures in production
- 99.9% uptime SLA
- API response time p95 < 500ms
- 100+ active users within 3 months
- 70%+ code coverage with comprehensive security testing

## Architecture

### System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Load Balancer (Railway/Nginx)               │
│                         - TLS termination                            │
│                         - Health checks                              │
└────────────────────────┬────────────────────────────────────────────┘
                         │
         ┌───────────────┴───────────────┐
         │                               │
┌────────▼────────┐            ┌────────▼────────┐
│  API Server 1   │            │  API Server N   │
│  (FastAPI)      │            │  (FastAPI)      │
│  - Auth         │            │  - Auth         │
│  - Rate Limit   │            │  - Rate Limit   │
│  - API Routes   │            │  - API Routes   │
└────────┬────────┘            └────────┬────────┘
         │                               │
         └───────────────┬───────────────┘
                         │
         ┌───────────────┴───────────────┐
         │                               │
┌────────▼────────┐            ┌────────▼────────┐
│  Redis Cache    │            │  PostgreSQL     │
│  - Sessions     │            │  - User data    │
│  - Rate limits  │            │  - Market data  │
│  - Cache        │            │  - Audit logs   │
└─────────────────┘            └────────┬────────┘
                                        │
                               ┌────────▼────────┐
                               │  Read Replicas  │
                               │  (optional)     │
                               └─────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                      Background Workers                              │
│  - Daily data ingestion (APScheduler)                               │
│  - YOLO pattern detection                                           │
│  - Report generation                                                │
│  - Webhook delivery                                                 │
│  - Backup service                                                   │
└────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                      External Services                               │
│  - Azure OpenAI (LLM)                                               │
│  - yfinance / Alpha Vantage (market data)                           │
│  - SMTP / Resend (email)                                            │
│  - S3 / Azure Blob (backups)                                        │
│  - Prometheus / Grafana (monitoring)                                │
└─────────────────────────────────────────────────────────────────────┘
```

### Architecture Principles

1. **Stateless API servers**: All session state in Redis, enabling horizontal scaling
2. **Defense in depth**: Multiple security layers (auth, rate limiting, input validation, output sanitization)
3. **Graceful degradation**: System continues with reduced functionality when non-critical services fail
4. **Multi-source redundancy**: Primary/secondary/fallback for all external data dependencies
5. **Observability first**: Comprehensive logging, metrics, and tracing from day one

### Migration from Current Architecture

**Current state** (single-process monolith):

- FastAPI + APScheduler in one process
- SQLite on persistent volume
- No Redis, no session management
- Single API key for all admin access

**Target state** (scalable multi-instance):

- Multiple FastAPI instances behind load balancer
- PostgreSQL with connection pooling
- Redis for sessions, cache, rate limits
- Per-user authentication with RBAC

**Migration strategy**:

1. Add PostgreSQL alongside SQLite (dual-write during transition)
2. Add Redis for new features (sessions, rate limits)
3. Refactor auth middleware to support both API key and JWT
4. Deploy second API instance to validate stateless operation
5. Migrate background jobs to separate worker process
6. Cut over to PostgreSQL as primary database
7. Remove SQLite support

## Components and Interfaces

### Authentication & Authorization Module

**Responsibilities**:

- User authentication (JWT tokens, API keys, email tokens)
- Role-based access control (admin, analyst, viewer)
- Session management
- Token expiration and revocation

**Interfaces**:

```python
class AuthService:
    def authenticate_api_key(self, api_key: str) -> User | None
    def authenticate_jwt(self, token: str) -> User | None
    def authenticate_email_token(self, token: str) -> User | None
    def create_jwt(self, user: User, expires_in: timedelta) -> str
    def create_email_token(self, user: User) -> str
    def revoke_token(self, token: str) -> bool
    def check_permission(self, user: User, resource: str, action: str) -> bool

class User:
    id: UUID
    username: str
    email: str
    password_hash: str
    role: UserRole  # admin | analyst | viewer
    created_at: datetime
    last_login: datetime
    is_active: bool

class UserRole(Enum):
    ADMIN = "admin"      # Full access
    ANALYST = "analyst"  # Read/write analysis, no admin
    VIEWER = "viewer"    # Read-only
```

**Security requirements**:

- Passwords hashed with bcrypt (12 rounds minimum)
- JWT tokens signed with HS256, 24-hour expiration
- Email tokens expire after 7 days
- Account lockout after 5 failed attempts (15-minute cooldown)
- All auth events logged to audit log

### Rate Limiting Module

**Responsibilities**:

- Per-IP rate limiting for public endpoints
- Per-user rate limiting for authenticated endpoints
- Sliding window algorithm for accurate rate calculation
- Rate limit bypass for admin users

**Interfaces**:

```python
class RateLimiter:
    def check_rate_limit(
        self,
        key: str,  # IP or user_id
        limit: int,
        window: timedelta
    ) -> RateLimitResult

class RateLimitResult:
    allowed: bool
    remaining: int
    reset_at: datetime
    retry_after: int | None  # seconds

# Configuration
RATE_LIMITS = {
    "public": (100, timedelta(minutes=1)),      # 100 req/min
    "authenticated": (1000, timedelta(hours=1)), # 1000 req/hour
    "export": (10, timedelta(hours=1)),         # 10 exports/hour
}
```

**Implementation**:

- Redis-backed sliding window counter
- Middleware intercepts all requests before route handlers
- Returns HTTP 429 with `Retry-After` header when exceeded
- Logs rate limit violations with IP, user, endpoint

### Audit Logging Module

**Responsibilities**:

- Immutable append-only audit trail
- Log all admin actions, auth events, data modifications
- Queryable by date, user, action type
- Export to external storage for long-term retention

**Interfaces**:

```python
class AuditLogger:
    def log_event(
        self,
        event_type: AuditEventType,
        user_id: UUID | None,
        resource: str,
        action: str,
        details: dict,
        ip_address: str,
        user_agent: str
    ) -> None

    def query_logs(
        self,
        start_date: datetime,
        end_date: datetime,
        user_id: UUID | None = None,
        event_type: AuditEventType | None = None
    ) -> list[AuditEvent]

class AuditEventType(Enum):
    AUTH_SUCCESS = "auth_success"
    AUTH_FAILURE = "auth_failure"
    API_REQUEST = "api_request"
    DATA_MODIFICATION = "data_modification"
    ADMIN_ACTION = "admin_action"
    EXPORT = "export"
    WEBHOOK_TRIGGER = "webhook_trigger"

class AuditEvent:
    id: UUID
    timestamp: datetime
    event_type: AuditEventType
    user_id: UUID | None
    resource: str
    action: str
    details: dict
    ip_address: str
    user_agent: str
    correlation_id: str  # For request tracing
```

**Storage**:

- Dedicated `audit_logs` table with indexed timestamp and user_id
- Partition by month for query performance
- Replicate to S3/Azure Blob daily for compliance (7-year retention)

### VIX Analysis Engine (Enhanced)

**Responsibilities**:

- Multi-source data fetching with fallback
- Adaptive compression threshold calculation
- Enhanced trap/reclaim pattern detection
- Key level source labeling
- VIX percentile prominence calculation

**Interfaces**:

```python
class VIXEngine:
    def fetch_vix_data(self) -> VIXData
    def calculate_term_structure(self) -> TermStructure
    def detect_compression(self, data: VIXData) -> CompressionSignal | None
    def detect_trap_reclaim(self, data: VIXData) -> list[TrapReclaimPattern]
    def calculate_key_levels(self, data: VIXData) -> list[KeyLevel]
    def calculate_percentile(self, data: VIXData) -> float

class VIXData:
    vix: pd.DataFrame  # OHLCV
    vix3m: pd.DataFrame | None
    vix6m: pd.DataFrame | None
    source: DataSource  # yfinance | alpha_vantage | csv_fallback
    timestamp: datetime

class TermStructure:
    vix_spot: float
    vix_3m: float | None
    vix_6m: float | None
    source: str  # "VIX3M" | "VIX6M" | "synthetic"
    contango: bool
    slope: float

class KeyLevel:
    price: float
    level_type: str  # "support" | "resistance"
    source: str  # "pivot_cluster" | "ma_anchor" | "fallback"
    tier: str  # "primary" | "secondary" | "fallback"
    confidence: float
    touches: int
    last_touch: datetime

class TrapReclaimPattern:
    pattern_type: str  # "bull_trap" | "bear_trap" | "support_reclaim" | "resistance_reclaim"
    date: datetime
    price: float
    confidence: float
    explanation: str
```

**Multi-source data fetching**:

```python
def fetch_vix_data(self) -> VIXData:
    # Try primary source
    try:
        data = self._fetch_yfinance()
        return VIXData(source=DataSource.YFINANCE, ...)
    except Exception as e:
        logger.warning(f"yfinance failed: {e}")

    # Try secondary source
    try:
        data = self._fetch_alpha_vantage()
        return VIXData(source=DataSource.ALPHA_VANTAGE, ...)
    except Exception as e:
        logger.warning(f"Alpha Vantage failed: {e}")

    # Fallback to local CSV
    data = self._load_csv_fallback()
    return VIXData(source=DataSource.CSV_FALLBACK, ...)
```

**Adaptive compression thresholds**:

```python
def calculate_compression_thresholds(self, vix_90d_percentile: float) -> tuple[float, float]:
    if vix_90d_percentile < 30:
        return (20, 80)  # Tight thresholds in low vol
    elif vix_90d_percentile < 70:
        return (25, 75)  # Moderate thresholds
    else:
        return (30, 70)  # Wide thresholds in high vol
```

### Webhook Delivery Module

**Responsibilities**:

- Webhook registration and management
- Event-triggered HTTP POST delivery
- Retry logic with exponential backoff
- Delivery status tracking and monitoring

**Interfaces**:

```python
class WebhookService:
    def register_webhook(
        self,
        user_id: UUID,
        url: str,
        events: list[WebhookEvent],
        auth_headers: dict | None = None,
        hmac_secret: str | None = None
    ) -> Webhook

    def trigger_webhook(
        self,
        event: WebhookEvent,
        payload: dict
    ) -> None

    def get_delivery_history(
        self,
        webhook_id: UUID,
        limit: int = 100
    ) -> list[WebhookDelivery]

class WebhookEvent(Enum):
    PATTERN_DETECTED = "pattern_detected"
    REGIME_CHANGE = "regime_change"
    ALERT_TRIGGERED = "alert_triggered"
    REPORT_GENERATED = "report_generated"

class Webhook:
    id: UUID
    user_id: UUID
    url: str
    events: list[WebhookEvent]
    auth_headers: dict
    hmac_secret: str | None
    is_active: bool
    created_at: datetime

class WebhookDelivery:
    id: UUID
    webhook_id: UUID
    event: WebhookEvent
    payload: dict
    status_code: int | None
    response_time_ms: int | None
    error: str | None
    attempts: int
    delivered_at: datetime | None
```

**Delivery logic**:

```python
async def _deliver_webhook(self, webhook: Webhook, payload: dict) -> None:
    headers = webhook.auth_headers.copy()

    # Add HMAC signature if configured
    if webhook.hmac_secret:
        signature = hmac.new(
            webhook.hmac_secret.encode(),
            json.dumps(payload).encode(),
            hashlib.sha256
        ).hexdigest()
        headers["X-Webhook-Signature"] = f"sha256={signature}"

    # Retry with exponential backoff
    for attempt in range(3):
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    webhook.url,
                    json=payload,
                    headers=headers
                )
                response.raise_for_status()
                # Log success
                return
        except Exception as e:
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)  # 1s, 2s, 4s
            else:
                # Log failure after 3 attempts
                logger.error(f"Webhook delivery failed: {e}")
```

### Backup & Restore Module

**Responsibilities**:

- Automated daily and incremental backups
- Backup integrity verification
- Point-in-time restore capability
- External storage integration (S3, Azure Blob)

**Interfaces**:

```python
class BackupService:
    def create_backup(self, backup_type: BackupType) -> Backup
    def list_backups(self, limit: int = 100) -> list[Backup]
    def restore_backup(self, backup_id: UUID) -> bool
    def verify_backup(self, backup_id: UUID) -> bool
    def cleanup_old_backups(self) -> int  # Returns count deleted

class BackupType(Enum):
    FULL = "full"
    INCREMENTAL = "incremental"

class Backup:
    id: UUID
    backup_type: BackupType
    size_bytes: int
    checksum: str
    storage_path: str
    created_at: datetime
    verified_at: datetime | None
    is_valid: bool
```

**Backup schedule**:

- Full backup: Daily at 00:00 UTC
- Incremental: Every 6 hours
- Retention: 30 days (daily), 90 days (weekly)

### Health Monitoring Module

**Responsibilities**:

- Prometheus metrics export
- Health check endpoint
- Dependency health tracking
- Alert generation for SLA violations

**Interfaces**:

```python
class HealthMonitor:
    def record_request(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        duration_ms: float
    ) -> None

    def record_db_query(self, query_type: str, duration_ms: float) -> None
    def record_llm_request(self, success: bool, tokens: int, duration_ms: float) -> None
    def record_yolo_detection(self, ticker: str, duration_ms: float) -> None
    def record_data_fetch(self, source: str, success: bool) -> None

    def get_health_status(self) -> HealthStatus
    def get_metrics(self) -> dict  # Prometheus format

class HealthStatus:
    status: str  # "healthy" | "degraded" | "unhealthy"
    uptime_seconds: float
    checks: dict[str, ComponentHealth]

class ComponentHealth:
    name: str
    status: str
    latency_ms: float | None
    error_rate: float
    last_check: datetime
```

**Metrics exposed**:

- `http_requests_total{endpoint, method, status}`
- `http_request_duration_seconds{endpoint, method}`
- `db_query_duration_seconds{query_type}`
- `llm_requests_total{success}`
- `llm_tokens_total`
- `yolo_detection_duration_seconds`
- `data_fetch_success_rate{source}`

## Data Models

### Database Schema Changes

#### New Tables

**users**

```sql
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(50) NOT NULL CHECK (role IN ('admin', 'analyst', 'viewer')),
    is_active BOOLEAN DEFAULT TRUE,
    failed_login_attempts INT DEFAULT 0,
    locked_until TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW(),
    last_login TIMESTAMP,
    INDEX idx_users_email (email),
    INDEX idx_users_username (username)
);
```

**sessions**

```sql
CREATE TABLE sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_hash VARCHAR(255) NOT NULL,
    expires_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    last_activity TIMESTAMP DEFAULT NOW(),
    ip_address VARCHAR(45),
    user_agent TEXT,
    INDEX idx_sessions_user_id (user_id),
    INDEX idx_sessions_token_hash (token_hash),
    INDEX idx_sessions_expires_at (expires_at)
);
```

**email_tokens**

```sql
CREATE TABLE email_tokens (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_hash VARCHAR(255) NOT NULL UNIQUE,
    expires_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    used_at TIMESTAMP,
    revoked_at TIMESTAMP,
    INDEX idx_email_tokens_token_hash (token_hash),
    INDEX idx_email_tokens_expires_at (expires_at)
);
```

**audit_logs**

```sql
CREATE TABLE audit_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp TIMESTAMP DEFAULT NOW(),
    event_type VARCHAR(50) NOT NULL,
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    resource VARCHAR(255),
    action VARCHAR(100),
    details JSONB,
    ip_address VARCHAR(45),
    user_agent TEXT,
    correlation_id UUID,
    INDEX idx_audit_logs_timestamp (timestamp),
    INDEX idx_audit_logs_user_id (user_id),
    INDEX idx_audit_logs_event_type (event_type),
    INDEX idx_audit_logs_correlation_id (correlation_id)
);

-- Partition by month for performance
CREATE TABLE audit_logs_2024_01 PARTITION OF audit_logs
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');
-- ... additional partitions
```

**webhooks**

```sql
CREATE TABLE webhooks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    url TEXT NOT NULL,
    events TEXT[] NOT NULL,  -- Array of event types
    auth_headers JSONB,
    hmac_secret VARCHAR(255),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    INDEX idx_webhooks_user_id (user_id),
    INDEX idx_webhooks_is_active (is_active)
);
```

**webhook_deliveries**

```sql
CREATE TABLE webhook_deliveries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    webhook_id UUID NOT NULL REFERENCES webhooks(id) ON DELETE CASCADE,
    event VARCHAR(50) NOT NULL,
    payload JSONB NOT NULL,
    status_code INT,
    response_time_ms INT,
    error TEXT,
    attempts INT DEFAULT 1,
    delivered_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW(),
    INDEX idx_webhook_deliveries_webhook_id (webhook_id),
    INDEX idx_webhook_deliveries_created_at (created_at)
);
```

**backups**

```sql
CREATE TABLE backups (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    backup_type VARCHAR(20) NOT NULL CHECK (backup_type IN ('full', 'incremental')),
    size_bytes BIGINT NOT NULL,
    checksum VARCHAR(64) NOT NULL,
    storage_path TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    verified_at TIMESTAMP,
    is_valid BOOLEAN DEFAULT TRUE,
    INDEX idx_backups_created_at (created_at),
    INDEX idx_backups_backup_type (backup_type)
);
```

**rate_limit_violations**

```sql
CREATE TABLE rate_limit_violations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    key VARCHAR(255) NOT NULL,  -- IP or user_id
    endpoint VARCHAR(255) NOT NULL,
    limit_type VARCHAR(50) NOT NULL,
    timestamp TIMESTAMP DEFAULT NOW(),
    INDEX idx_rate_limit_violations_key (key),
    INDEX idx_rate_limit_violations_timestamp (timestamp)
);
```

#### Modified Tables

**yolo_patterns** (add source labeling)

```sql
ALTER TABLE yolo_patterns
ADD COLUMN detection_source VARCHAR(50) DEFAULT 'yolo',
ADD COLUMN model_version VARCHAR(50),
ADD COLUMN preprocessing_params JSONB;
```

**price_daily** (add data source tracking)

```sql
ALTER TABLE price_daily
ADD COLUMN data_source VARCHAR(50) DEFAULT 'yfinance',
ADD COLUMN fetch_timestamp TIMESTAMP DEFAULT NOW();

CREATE INDEX idx_price_daily_data_source ON price_daily(data_source);
```

### Redis Data Structures

**Session storage**

```
Key: session:{token_hash}
Value: {user_id, role, expires_at, ip_address}
TTL: 24 hours
```

**Rate limiting**

```
Key: ratelimit:{key}:{endpoint}:{window}
Value: Sorted set of timestamps
TTL: window duration + 1 minute
```

**Cache**

```
Key: cache:dashboard:{ticker}
Value: JSON serialized dashboard data
TTL: 5 minutes

Key: cache:vix_analysis
Value: JSON serialized VIX analysis
TTL: 5 minutes
```

### API Response Models

**Dashboard Response**

```json
{
  "ticker": "AAPL",
  "timestamp": "2024-01-15T10:30:00Z",
  "price_data": {
    "current": 185.5,
    "change": 2.3,
    "change_pct": 1.25,
    "volume": 52000000
  },
  "patterns": [
    {
      "id": "uuid",
      "type": "bull_flag",
      "confidence": 0.85,
      "source": "hybrid",
      "date_range": ["2024-01-01", "2024-01-15"],
      "price_range": [180.0, 186.0],
      "explanation": "Strong uptrend followed by consolidation"
    }
  ],
  "levels": [
    {
      "price": 182.5,
      "type": "support",
      "source": "pivot_cluster",
      "tier": "primary",
      "confidence": 0.92,
      "touches": 5
    }
  ],
  "vix_context": {
    "current": 14.5,
    "percentile": 35.2,
    "regime": "low_volatility",
    "term_structure": {
      "spot": 14.5,
      "three_month": 16.2,
      "source": "VIX3M",
      "contango": true
    }
  },
  "data_sources": {
    "price": "yfinance",
    "vix": "yfinance",
    "patterns": ["hybrid", "yolo_daily", "yolo_weekly"]
  }
}
```

**Audit Log Query Response**

```json
{
  "logs": [
    {
      "id": "uuid",
      "timestamp": "2024-01-15T10:30:00Z",
      "event_type": "admin_action",
      "user": {
        "id": "uuid",
        "username": "admin_user",
        "role": "admin"
      },
      "resource": "/api/admin/users",
      "action": "create_user",
      "details": {
        "new_user_id": "uuid",
        "new_user_role": "analyst"
      },
      "ip_address": "192.168.1.100",
      "correlation_id": "uuid"
    }
  ],
  "pagination": {
    "total": 1523,
    "page": 1,
    "per_page": 50,
    "pages": 31
  }
}
```

### Configuration Management

**Environment Variables**

```bash
# Database
DATABASE_URL=postgresql://user:pass@host:5432/trader_koo
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=10

# Redis
REDIS_URL=redis://host:6379/0
REDIS_POOL_SIZE=10

# Authentication
JWT_SECRET_KEY=<random-256-bit-key>
JWT_ALGORITHM=HS256
JWT_EXPIRATION_HOURS=24
EMAIL_TOKEN_EXPIRATION_DAYS=7

# API Keys (legacy support)
TRADER_KOO_API_KEY=<legacy-admin-key>
ADMIN_STRICT_API_KEY=1

# Rate Limiting
RATE_LIMIT_PUBLIC_PER_MINUTE=100
RATE_LIMIT_AUTH_PER_HOUR=1000
RATE_LIMIT_EXPORT_PER_HOUR=10

# External Services
AZURE_OPENAI_ENDPOINT=https://...
AZURE_OPENAI_API_KEY=<key>
AZURE_OPENAI_DEPLOYMENT=gpt-4
ALPHA_VANTAGE_API_KEY=<key>

# SMTP
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_SECURITY=starttls
SMTP_USER=<email>
SMTP_PASS=<app-password>

# Backup
BACKUP_STORAGE_TYPE=s3  # s3 | azure_blob | local
BACKUP_S3_BUCKET=trader-koo-backups
BACKUP_S3_REGION=us-east-1
AWS_ACCESS_KEY_ID=<key>
AWS_SECRET_ACCESS_KEY=<secret>

# Monitoring
PROMETHEUS_ENABLED=true
PROMETHEUS_PORT=9090
LOG_LEVEL=INFO
SENTRY_DSN=<optional>

# Feature Flags
ENABLE_WEBHOOKS=true
ENABLE_SOCIAL_FEATURES=false
ENABLE_REALTIME_WEBSOCKET=false
```

**Configuration Validation**

```python
class Config:
    """Application configuration with validation"""

    def __init__(self):
        self.validate_required()
        self.validate_security()
        self.validate_connections()

    def validate_required(self):
        """Ensure all required env vars are set"""
        required = [
            "DATABASE_URL",
            "REDIS_URL",
            "JWT_SECRET_KEY",
        ]

        if os.getenv("ADMIN_STRICT_API_KEY", "1") == "1":
            required.append("TRADER_KOO_API_KEY")

        missing = [var for var in required if not os.getenv(var)]
        if missing:
            raise ConfigError(f"Missing required env vars: {missing}")

    def validate_security(self):
        """Validate security-related configuration"""
        api_key = os.getenv("TRADER_KOO_API_KEY", "")
        if api_key and len(api_key) < 32:
            raise ConfigError("TRADER_KOO_API_KEY must be at least 32 characters")

        jwt_secret = os.getenv("JWT_SECRET_KEY", "")
        if len(jwt_secret) < 32:
            raise ConfigError("JWT_SECRET_KEY must be at least 32 characters")

    def validate_connections(self):
        """Test critical service connections"""
        try:
            # Test database connection
            engine = create_engine(self.database_url)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
        except Exception as e:
            raise ConfigError(f"Database connection failed: {e}")

        try:
            # Test Redis connection
            redis_client = redis.from_url(self.redis_url)
            redis_client.ping()
        except Exception as e:
            raise ConfigError(f"Redis connection failed: {e}")
```

## Correctness Properties

_A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees._

### Security Properties

### Property 1: API Key Length Validation

_For any_ API key string provided as `TRADER_KOO_API_KEY`, if the key is non-empty, then its length must be at least 32 characters, otherwise the platform should reject it during startup validation.

**Validates: Requirements 1.5**

### Property 2: LLM Output Schema Validation with Fallback

_For any_ LLM response, the platform should validate it against the expected JSON schema, and if validation fails (missing required fields, wrong types, or length violations), then the platform should log the validation error with context and fall back to deterministic rule-based output.

**Validates: Requirements 2.1, 2.2, 2.6, 2.7, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7**

### Property 3: LLM Output Sanitization

_For any_ LLM-generated text, the platform should strip HTML tags and script content before storage, truncate fields exceeding maximum length with ellipsis, and escape special characters before rendering in HTML contexts.

**Validates: Requirements 2.3, 2.4, 2.5**

### Property 4: Email Token Expiration

_For any_ email authentication token, the platform should set expiration to exactly 7 days from creation, store the creation timestamp, and reject any authentication attempt where the token age exceeds 7 days.

**Validates: Requirements 3.1, 3.2, 3.3, 3.4**

### Property 5: Email Token Revocation

_For any_ email token, an admin should be able to revoke it before expiration, and after revocation, any authentication attempt with that token should be rejected.

**Validates: Requirements 3.6**

### Property 6: CORS Origin Validation

_For any_ origin string in the CORS configuration, the platform should validate it matches the pattern `https://` or `http://localhost`, and for any incoming CORS request, the platform should reject requests from origins not in the allowed list and log the rejection.

**Validates: Requirements 4.2, 4.3, 4.4, 4.5**

### Property 7: CORS Credentials Header

_For any_ HTTP response, the platform should include the header `Access-Control-Allow-Credentials: false` by default.

**Validates: Requirements 4.7**

### Property 8: Secret Redaction in Logs

_For any_ data structure being logged, if it contains keys matching secret patterns (API keys, passwords, tokens), then the platform should replace those values with `[REDACTED]` in the log output.

**Validates: Requirements 6.2, 6.3**

### Property 9: Secret Redaction in Error Responses

_For any_ error response, the platform should strip environment variables and configuration values from stack traces before returning to the client.

**Validates: Requirements 6.4**

### Property 10: VIX Trap and Reclaim Labeling

_For any_ VIX price pattern where VIX breaks a level then reverses, the platform should correctly label it as: "bull_trap" or "failed_breakout" (breaks resistance, reverses down), "bear_trap" or "failed_breakdown" (breaks support, reverses up), "support_reclaim" (reclaims support after breakdown), or "resistance_reclaim" (reclaims resistance after breakout).

**Validates: Requirements 8.1, 8.2, 8.3, 8.4**

### Property 11: Term Structure Source Labeling

_For any_ term structure calculation, the platform should label the data source ("VIX3M", "VIX6M", or "synthetic"), log the source and availability, include the source in all displays, and include the data source timestamp.

**Validates: Requirements 9.3, 9.4, 9.6**

### Property 12: Key Level Source Labeling

_For any_ VIX key level, the platform should label it with its source ("pivot_cluster", "ma_anchor", or "fallback"), include the source label in UI displays, include a "source" field in API responses, and include the source in report narratives.

**Validates: Requirements 10.1, 10.2, 10.3, 10.6**

### Property 13: Key Level Prioritization

_For any_ set of key levels, the platform should prioritize them in order: pivot_cluster levels first, then ma_anchor levels, then fallback levels.

**Validates: Requirements 10.4**

### Property 14: VIX Percentile Calculation

_For any_ VIX dataset, the platform should calculate the percentile rank over a 252-day rolling window and include it as a primary factor in health score calculation.

**Validates: Requirements 11.1, 11.4**

### Property 15: VIX Percentile Color Coding

_For any_ VIX percentile value, the platform should color-code it as: green for 0-30, yellow for 30-70, and red for 70-100.

**Validates: Requirements 11.3**

### Property 16: VIX Percentile in API Responses

_For any_ regime analysis API response, the platform should include the VIX percentile value.

**Validates: Requirements 11.6**

### Property 17: Data Source Logging

_For any_ ticker data fetch, the platform should log which data source was used (yfinance, Alpha Vantage, or CSV fallback).

**Validates: Requirements 12.4**

### Property 18: Data Source in API Responses

_For any_ API response containing market data, the platform should include the data source and timestamp.

**Validates: Requirements 12.7**

### Property 19: Adaptive Compression Thresholds

_For any_ VIX 90-day percentile value, the platform should calculate compression thresholds dynamically: (20, 80) for percentile < 30, (25, 75) for percentile 30-70, and (30, 70) for percentile > 70.

**Validates: Requirements 13.1, 13.2, 13.3, 13.4**

### Property 20: Compression Signal Labeling

_For any_ compression signal, the platform should label it with the threshold regime (tight, moderate, or wide).

**Validates: Requirements 13.6**

### Property 21: Bull Trap Detection

_For any_ VIX price sequence where VIX breaks above resistance then closes below resistance within 3 bars, the platform should detect it as a bull trap and calculate confidence based on volume profile and reversal speed.

**Validates: Requirements 14.1, 14.3**

### Property 22: Bear Trap Detection

_For any_ VIX price sequence where VIX breaks below support then closes above support within 3 bars, the platform should detect it as a bear trap and calculate confidence based on volume profile and reversal speed.

**Validates: Requirements 14.2, 14.3**

### Property 23: Support Reclaim Detection

_For any_ VIX price sequence where VIX closes above broken support for 2 or more consecutive bars, the platform should detect it as a support reclaim.

**Validates: Requirements 14.4**

### Property 24: Resistance Reclaim Detection

_For any_ VIX price sequence where VIX closes below broken resistance for 2 or more consecutive bars, the platform should detect it as a resistance reclaim.

**Validates: Requirements 14.5**

### Property 25: Trap/Reclaim in Regime Analysis

_For any_ regime analysis output, the platform should include detected trap/reclaim patterns with confidence scores.

**Validates: Requirements 14.6**

### Property 26: Trap/Reclaim Visual Display

_For any_ trap/reclaim pattern, the platform should display it on the VIX chart with distinct visual markers.

**Validates: Requirements 14.7**

### Enterprise Feature Properties

### Property 27: Audit Log Completeness

_For any_ admin API request, authentication attempt, or data modification, the platform should create an audit log entry with timestamp, user, resource, action, details, IP address, and user agent.

**Validates: Requirements 15.1, 15.2, 15.3**

### Property 28: Audit Log Immutability

_For any_ audit log entry, once created, it should not be possible to update or delete it (append-only).

**Validates: Requirements 15.4**

### Property 29: Audit Log Querying

_For any_ audit log query with date range, user, and/or event type filters, the platform should return matching log entries.

**Validates: Requirements 15.6**

### Property 30: JWT Token with Role Claim

_For any_ successful user authentication, the platform should issue a JWT token that includes the user's role as a claim.

**Validates: Requirements 16.3**

### Property 31: Role-Based Access Control

_For any_ protected endpoint and user with a specific role, the platform should enforce permissions: admin gets full access, analyst gets read/write to analysis/reports but no admin access, viewer gets read-only access to dashboards/reports.

**Validates: Requirements 16.4, 16.5, 16.6, 16.7**

### Property 32: Role Change Audit Logging

_For any_ user role change, the platform should log the change to the audit log.

**Validates: Requirements 16.9**

### Property 33: Password Complexity Validation

_For any_ password string, the platform should validate it meets complexity requirements: minimum 12 characters, at least one uppercase, one lowercase, one number, and one special character.

**Validates: Requirements 16.10**

### Property 34: Rate Limiting Enforcement

_For any_ API request, the platform should enforce rate limits based on the request type: 100 requests per minute for public endpoints (per IP), 1000 requests per hour for authenticated endpoints (per user), and return HTTP 429 with Retry-After header when exceeded.

**Validates: Requirements 17.1, 17.2, 17.3**

### Property 35: Rate Limit Sliding Window

_For any_ rate limit calculation, the platform should use a sliding window algorithm to accurately count requests within the time window.

**Validates: Requirements 17.4**

### Property 36: Rate Limit Violation Logging

_For any_ rate limit violation, the platform should log it with IP address, user ID, and endpoint.

**Validates: Requirements 17.8**

### Property 37: Webhook Event Triggering

_For any_ platform event (pattern_detected, regime_change, alert_triggered, report_generated), the platform should trigger all registered webhooks subscribed to that event type.

**Validates: Requirements 18.2**

### Property 38: Webhook Payload Structure

_For any_ webhook delivery, the platform should POST a JSON payload containing event type, timestamp, and event-specific data to the registered URL.

**Validates: Requirements 18.3, 18.4**

### Property 39: Webhook Retry Logic

_For any_ failed webhook delivery, the platform should retry up to 3 times with exponential backoff (1s, 2s, 4s).

**Validates: Requirements 18.5**

### Property 40: Webhook Delivery Logging

_For any_ webhook delivery attempt, the platform should log it with status code, response time, and any error message.

**Validates: Requirements 18.6**

### Property 41: Webhook URL Validation

_For any_ webhook URL, the platform should validate it uses HTTPS (except for localhost URLs which can use HTTP for testing).

**Validates: Requirements 18.9**

### Property 42: Webhook Timeout

_For any_ webhook HTTP request, the platform should timeout after 10 seconds.

**Validates: Requirements 18.10**

### Security Properties

### Property 43: CSRF Protection

_For any_ state-changing operation (POST, PUT, DELETE), the platform should validate CSRF tokens to prevent cross-site request forgery attacks.

**Validates: Requirements 47.2**

### Property 44: SQL Injection Prevention

_For any_ user input used in database queries, the platform should use parameterized queries or sanitize the input to prevent SQL injection attacks.

**Validates: Requirements 47.3**

### Property 45: XSS Prevention

_For any_ user input that will be rendered in HTML, the platform should sanitize it to prevent cross-site scripting attacks.

**Validates: Requirements 47.4**

### Property 46: Request Size Limits

_For any_ HTTP request, the platform should enforce a maximum size limit of 10MB and reject requests exceeding this limit.

**Validates: Requirements 47.5**

### Property 47: Request Timeout Limits

_For any_ HTTP request, the platform should enforce a maximum timeout of 30 seconds.

**Validates: Requirements 47.6**

### Property 48: Password Hashing

_For any_ password being stored, the platform should hash it using bcrypt with a minimum of 12 rounds.

**Validates: Requirements 47.7**

### Property 49: Security Event Logging

_For any_ security event (failed authentication, rate limit violation, suspicious activity), the platform should log it with relevant context.

**Validates: Requirements 47.9**

## Error Handling

### Error Handling Strategy

The platform implements a layered error handling approach with graceful degradation:

1. **Input Validation Layer**: Catch invalid input at API boundary, return 400 Bad Request
2. **Business Logic Layer**: Handle domain-specific errors, return appropriate 4xx codes
3. **External Service Layer**: Implement fallbacks for external service failures
4. **Infrastructure Layer**: Handle database/Redis failures with circuit breakers

### Error Response Format

All error responses follow a consistent JSON structure:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input parameters",
    "details": {
      "field": "email",
      "reason": "Invalid email format"
    },
    "correlation_id": "uuid",
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

### Error Categories

**Client Errors (4xx)**:

- `400 Bad Request`: Invalid input, validation failures
- `401 Unauthorized`: Missing or invalid authentication
- `403 Forbidden`: Insufficient permissions
- `404 Not Found`: Resource does not exist
- `429 Too Many Requests`: Rate limit exceeded

**Server Errors (5xx)**:

- `500 Internal Server Error`: Unexpected server error
- `502 Bad Gateway`: External service failure (with fallback attempted)
- `503 Service Unavailable`: System overloaded or maintenance
- `504 Gateway Timeout`: External service timeout

### Graceful Degradation

**LLM Service Failure**:

```python
try:
    narrative = await llm_service.generate_narrative(data)
except LLMServiceError as e:
    logger.error(f"LLM service failed: {e}", extra={"correlation_id": correlation_id})
    # Fall back to deterministic template
    narrative = generate_template_narrative(data)
    metrics.increment("llm_fallback_count")
```

**Data Source Failure**:

```python
async def fetch_market_data(ticker: str) -> MarketData:
    # Try primary source
    try:
        return await yfinance_client.fetch(ticker)
    except YFinanceError as e:
        logger.warning(f"yfinance failed for {ticker}: {e}")
        metrics.increment("yfinance_failure_count")

    # Try secondary source
    try:
        return await alpha_vantage_client.fetch(ticker)
    except AlphaVantageError as e:
        logger.warning(f"Alpha Vantage failed for {ticker}: {e}")
        metrics.increment("alpha_vantage_failure_count")

    # Fall back to cached/CSV data
    return load_fallback_data(ticker)
```

**Database Failure**:

```python
@circuit_breaker(failure_threshold=5, timeout=60)
async def query_database(query: str) -> list[dict]:
    try:
        return await db.execute(query)
    except DatabaseError as e:
        logger.error(f"Database query failed: {e}")
        # Circuit breaker will open after 5 failures
        # Return cached data if available
        if cache_key in redis:
            return redis.get(cache_key)
        raise ServiceUnavailableError("Database temporarily unavailable")
```

### Circuit Breaker Pattern

Implement circuit breakers for all external dependencies:

```python
class CircuitBreaker:
    def __init__(self, failure_threshold: int, timeout: int):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed | open | half_open

    def call(self, func, *args, **kwargs):
        if self.state == "open":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "half_open"
            else:
                raise CircuitBreakerOpenError("Circuit breaker is open")

        try:
            result = func(*args, **kwargs)
            if self.state == "half_open":
                self.state = "closed"
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
            raise
```

### Retry Logic

Implement exponential backoff for transient failures:

```python
async def retry_with_backoff(
    func: Callable,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0
) -> Any:
    for attempt in range(max_attempts):
        try:
            return await func()
        except TransientError as e:
            if attempt == max_attempts - 1:
                raise

            delay = min(base_delay * (2 ** attempt), max_delay)
            logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay}s: {e}")
            await asyncio.sleep(delay)
```

### Error Logging

All errors are logged with structured context:

```python
logger.error(
    "Failed to process dashboard request",
    extra={
        "correlation_id": correlation_id,
        "user_id": user_id,
        "ticker": ticker,
        "error_type": type(e).__name__,
        "error_message": str(e),
        "stack_trace": traceback.format_exc(),
        "request_path": request.url.path,
        "request_method": request.method,
    }
)
```

### Secret Sanitization in Errors

Before logging or returning errors, sanitize sensitive data:

```python
def sanitize_error_context(context: dict) -> dict:
    """Remove secrets from error context before logging"""
    secret_patterns = [
        "api_key", "password", "token", "secret", "credential",
        "authorization", "cookie", "session"
    ]

    sanitized = {}
    for key, value in context.items():
        if any(pattern in key.lower() for pattern in secret_patterns):
            sanitized[key] = "[REDACTED]"
        elif isinstance(value, dict):
            sanitized[key] = sanitize_error_context(value)
        else:
            sanitized[key] = value

    return sanitized
```

## Testing Strategy

### Dual Testing Approach

The platform requires both unit testing and property-based testing for comprehensive coverage:

- **Unit tests**: Verify specific examples, edge cases, error conditions, and integration points
- **Property tests**: Verify universal properties across all inputs through randomization
- Both are complementary and necessary - unit tests catch concrete bugs, property tests verify general correctness

### Property-Based Testing

**Library Selection**: Use `hypothesis` for Python property-based testing

**Configuration**: Each property test runs minimum 100 iterations due to randomization

**Test Tagging**: Each property test references its design document property:

```python
@given(api_key=st.text(min_size=1, max_size=100))
def test_api_key_length_validation(api_key):
    """
    Feature: enterprise-platform-upgrade, Property 1: API Key Length Validation

    For any API key string provided as TRADER_KOO_API_KEY, if the key is non-empty,
    then its length must be at least 32 characters.
    """
    if len(api_key) < 32:
        with pytest.raises(ConfigError):
            validate_api_key(api_key)
    else:
        # Should not raise
        validate_api_key(api_key)
```

**Property Test Examples**:

```python
# Property 2: LLM Output Schema Validation
@given(llm_output=st.dictionaries(
    keys=st.text(),
    values=st.one_of(st.text(), st.integers(), st.booleans())
))
@settings(max_examples=100)
def test_llm_output_validation_with_fallback(llm_output):
    """
    Feature: enterprise-platform-upgrade, Property 2: LLM Output Schema Validation with Fallback
    """
    result = process_llm_output(llm_output)

    # Should always return valid output (either validated or fallback)
    assert result is not None
    assert validate_output_schema(result)

    # If input was invalid, should have logged and used fallback
    if not validate_llm_schema(llm_output):
        assert "LLM validation failed" in caplog.text
        assert result == get_fallback_output()

# Property 4: Email Token Expiration
@given(
    creation_time=st.datetimes(
        min_value=datetime(2020, 1, 1),
        max_value=datetime(2030, 1, 1)
    ),
    auth_time=st.datetimes(
        min_value=datetime(2020, 1, 1),
        max_value=datetime(2030, 1, 1)
    )
)
@settings(max_examples=100)
def test_email_token_expiration(creation_time, auth_time):
    """
    Feature: enterprise-platform-upgrade, Property 4: Email Token Expiration
    """
    token = create_email_token(user_id="test", created_at=creation_time)

    # Token should have 7-day expiration
    assert token.expires_at == creation_time + timedelta(days=7)

    # Authentication should succeed if within 7 days, fail otherwise
    age = auth_time - creation_time
    result = authenticate_email_token(token.token, current_time=auth_time)

    if age <= timedelta(days=7):
        assert result.success
    else:
        assert not result.success
        assert "expired" in result.error.lower()

# Property 10: VIX Trap and Reclaim Labeling
@given(
    vix_prices=st.lists(
        st.floats(min_value=10.0, max_value=50.0),
        min_size=10,
        max_size=100
    ),
    resistance_level=st.floats(min_value=15.0, max_value=45.0)
)
@settings(max_examples=100)
def test_vix_trap_labeling(vix_prices, resistance_level):
    """
    Feature: enterprise-platform-upgrade, Property 10: VIX Trap and Reclaim Labeling
    """
    # Create a bull trap pattern: break above resistance, then reverse
    pattern_prices = vix_prices.copy()
    # Insert breakout
    pattern_prices.append(resistance_level + 1.0)
    # Insert reversal within 3 bars
    pattern_prices.extend([resistance_level - 0.5, resistance_level - 1.0])

    patterns = detect_vix_patterns(pattern_prices, resistance=resistance_level)

    # Should detect as bull_trap or failed_breakout
    trap_patterns = [p for p in patterns if p.type in ["bull_trap", "failed_breakout"]]
    assert len(trap_patterns) > 0

# Property 34: Rate Limiting Enforcement
@given(
    request_count=st.integers(min_value=1, max_value=200),
    time_window=st.integers(min_value=1, max_value=120)  # seconds
)
@settings(max_examples=100)
def test_rate_limiting_enforcement(request_count, time_window):
    """
    Feature: enterprise-platform-upgrade, Property 34: Rate Limiting Enforcement
    """
    rate_limiter = RateLimiter()
    ip_address = "192.168.1.100"

    # Public endpoint: 100 req/min
    limit = 100
    window = 60

    results = []
    for i in range(request_count):
        result = rate_limiter.check_rate_limit(
            key=ip_address,
            limit=limit,
            window=timedelta(seconds=window)
        )
        results.append(result)

    # First 100 requests should be allowed
    assert all(r.allowed for r in results[:limit])

    # Requests beyond limit should be denied with 429
    if request_count > limit:
        assert not results[limit].allowed
        assert results[limit].retry_after is not None
```

### Unit Testing

**Test Organization**:

```
tests/
├── unit/
│   ├── test_auth.py
│   ├── test_rate_limiting.py
│   ├── test_audit_logging.py
│   ├── test_vix_engine.py
│   ├── test_webhooks.py
│   └── test_security.py
├── integration/
│   ├── test_dashboard_flow.py
│   ├── test_daily_update_pipeline.py
│   ├── test_yolo_detection.py
│   └── test_multi_user_auth.py
├── property/
│   ├── test_security_properties.py
│   ├── test_vix_properties.py
│   └── test_enterprise_properties.py
└── conftest.py
```

**Unit Test Examples**:

```python
# Example-based tests for specific scenarios
def test_startup_fails_without_api_key_in_strict_mode():
    """Test that platform refuses to start when API key is missing in strict mode"""
    os.environ["ADMIN_STRICT_API_KEY"] = "1"
    os.environ.pop("TRADER_KOO_API_KEY", None)

    with pytest.raises(ConfigError, match="TRADER_KOO_API_KEY is required"):
        config = Config()

def test_vix_term_structure_fallback_to_vix6m():
    """Test that VIX engine falls back to VIX6M when VIX3M unavailable"""
    # Mock VIX3M as unavailable
    with patch("yfinance.download", side_effect=Exception("VIX3M not found")):
        # Mock VIX6M as available
        with patch("alpha_vantage.fetch", return_value=mock_vix6m_data):
            term_structure = vix_engine.calculate_term_structure()

            assert term_structure.source == "VIX6M"
            assert term_structure.vix_6m is not None

def test_account_lockout_after_5_failed_attempts():
    """Test that account is locked after 5 failed login attempts"""
    user = create_test_user()

    # Attempt 5 failed logins
    for i in range(5):
        result = auth_service.authenticate(user.username, "wrong_password")
        assert not result.success

    # 6th attempt should be blocked due to lockout
    result = auth_service.authenticate(user.username, "correct_password")
    assert not result.success
    assert "account locked" in result.error.lower()

    # Check user is locked
    user = db.get_user(user.id)
    assert user.locked_until is not None
    assert user.locked_until > datetime.now()

def test_webhook_hmac_signature():
    """Test that webhook HMAC signatures are correctly generated and validated"""
    webhook = Webhook(
        url="https://example.com/webhook",
        hmac_secret="test_secret_key_12345"
    )

    payload = {"event": "pattern_detected", "ticker": "AAPL"}

    # Generate signature
    signature = generate_webhook_signature(webhook.hmac_secret, payload)

    # Signature should be valid
    assert validate_webhook_signature(webhook.hmac_secret, payload, signature)

    # Modified payload should fail validation
    payload["ticker"] = "MSFT"
    assert not validate_webhook_signature(webhook.hmac_secret, payload, signature)
```

### Integration Testing

**Integration test scenarios**:

```python
@pytest.mark.integration
async def test_complete_dashboard_flow():
    """Test end-to-end dashboard data flow"""
    # 1. Authenticate user
    token = await auth_service.create_jwt(test_user)

    # 2. Request dashboard data
    response = await client.get(
        "/api/dashboard/AAPL",
        headers={"Authorization": f"Bearer {token}"}
    )

    assert response.status_code == 200
    data = response.json()

    # 3. Verify data structure
    assert "ticker" in data
    assert "patterns" in data
    assert "levels" in data
    assert "vix_context" in data

    # 4. Verify data sources are labeled
    assert "data_sources" in data
    assert data["data_sources"]["price"] in ["yfinance", "alpha_vantage", "csv_fallback"]

    # 5. Verify audit log entry created
    logs = await audit_logger.query_logs(user_id=test_user.id)
    assert any(log.resource == "/api/dashboard/AAPL" for log in logs)

@pytest.mark.integration
async def test_webhook_delivery_with_retry():
    """Test webhook delivery with retry logic"""
    # 1. Register webhook
    webhook = await webhook_service.register_webhook(
        user_id=test_user.id,
        url="https://httpstat.us/500",  # Will fail
        events=[WebhookEvent.PATTERN_DETECTED]
    )

    # 2. Trigger event
    await webhook_service.trigger_webhook(
        event=WebhookEvent.PATTERN_DETECTED,
        payload={"ticker": "AAPL", "pattern": "bull_flag"}
    )

    # 3. Wait for retries to complete
    await asyncio.sleep(10)

    # 4. Verify 3 delivery attempts were made
    deliveries = await webhook_service.get_delivery_history(webhook.id)
    assert len(deliveries) == 1
    assert deliveries[0].attempts == 3
    assert deliveries[0].status_code == 500
```

### Security Testing

**Security test requirements**:

```python
def test_admin_endpoint_authentication():
    """Test that all admin endpoints require authentication"""
    admin_endpoints = [
        "/api/admin/users",
        "/api/admin/run-yolo-seed",
        "/api/admin/trigger-update",
        "/api/admin/yolo-status",
    ]

    for endpoint in admin_endpoints:
        # Without API key should return 401
        response = client.get(endpoint)
        assert response.status_code == 401

        # With invalid API key should return 401
        response = client.get(
            endpoint,
            headers={"X-API-Key": "invalid_key"}
        )
        assert response.status_code == 401

        # With valid API key should succeed
        response = client.get(
            endpoint,
            headers={"X-API-Key": valid_api_key}
        )
        assert response.status_code in [200, 404]  # 404 if endpoint doesn't support GET

def test_secret_redaction_in_logs(caplog):
    """Test that secrets are redacted in all log output"""
    secrets = {
        "TRADER_KOO_API_KEY": "super_secret_key_12345",
        "JWT_SECRET_KEY": "jwt_secret_67890",
        "AZURE_OPENAI_API_KEY": "azure_key_abcdef"
    }

    # Log data containing secrets
    logger.info("Config loaded", extra={"config": secrets})

    # Verify secrets are redacted in log output
    for secret_value in secrets.values():
        assert secret_value not in caplog.text

    assert "[REDACTED]" in caplog.text

def test_sql_injection_prevention():
    """Test that SQL injection attempts are prevented"""
    malicious_inputs = [
        "'; DROP TABLE users; --",
        "1' OR '1'='1",
        "admin'--",
        "1; DELETE FROM audit_logs WHERE 1=1; --"
    ]

    for malicious_input in malicious_inputs:
        # Should not execute malicious SQL
        result = db.query_user_by_username(malicious_input)
        assert result is None

        # Database should still be intact
        assert db.table_exists("users")
        assert db.table_exists("audit_logs")

def test_xss_prevention():
    """Test that XSS attempts are sanitized"""
    xss_inputs = [
        "<script>alert('XSS')</script>",
        "<img src=x onerror=alert('XSS')>",
        "javascript:alert('XSS')",
        "<iframe src='javascript:alert(\"XSS\")'></iframe>"
    ]

    for xss_input in xss_inputs:
        # Submit as user input
        response = client.post(
            "/api/analysis/comment",
            json={"text": xss_input},
            headers={"Authorization": f"Bearer {token}"}
        )

        # Retrieve and verify sanitized
        response = client.get("/api/analysis/comments")
        html = response.text

        # Should not contain executable script
        assert "<script>" not in html
        assert "javascript:" not in html
        assert "onerror=" not in html
```

### Performance Testing

**Load testing with Locust**:

```python
from locust import HttpUser, task, between

class TradingPlatformUser(HttpUser):
    wait_time = between(1, 3)

    def on_start(self):
        # Authenticate
        response = self.client.post("/api/auth/login", json={
            "username": "test_user",
            "password": "test_password"
        })
        self.token = response.json()["token"]

    @task(3)
    def view_dashboard(self):
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        ticker = random.choice(tickers)
        self.client.get(
            f"/api/dashboard/{ticker}",
            headers={"Authorization": f"Bearer {self.token}"}
        )

    @task(1)
    def view_vix_analysis(self):
        self.client.get(
            "/api/vix/analysis",
            headers={"Authorization": f"Bearer {self.token}"}
        )

    @task(1)
    def query_audit_logs(self):
        self.client.get(
            "/api/admin/audit-logs",
            headers={"X-API-Key": admin_api_key}
        )

# Run: locust -f tests/performance/test_load.py --users 100 --spawn-rate 10
```

### Test Coverage Requirements

- Overall code coverage: minimum 70%
- Security-critical code: 100% coverage
- P0 requirements: 100% test coverage
- P1 requirements: 90% test coverage
- P2/P3 requirements: 70% test coverage

### Continuous Integration

**GitHub Actions workflow**:

```yaml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: postgres
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

      redis:
        image: redis:7
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt

      - name: Run linting
        run: |
          flake8 trader_koo/
          black --check trader_koo/
          mypy trader_koo/

      - name: Run security scanning
        run: |
          bandit -r trader_koo/
          safety check

      - name: Run unit tests
        run: |
          pytest tests/unit/ -v --cov=trader_koo --cov-report=xml

      - name: Run property tests
        run: |
          pytest tests/property/ -v --hypothesis-show-statistics

      - name: Run integration tests
        run: |
          pytest tests/integration/ -v

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml

      - name: Check coverage threshold
        run: |
          coverage report --fail-under=70
```

## Deployment Architecture

### Production Deployment Topology

```
┌─────────────────────────────────────────────────────────────┐
│                     Cloudflare / CDN                         │
│                     - DDoS protection                        │
│                     - TLS termination                        │
│                     - Static asset caching                   │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                   Load Balancer (Railway)                    │
│                   - Health checks                            │
│                   - SSL/TLS                                  │
│                   - Request routing                          │
└────────────────────────┬────────────────────────────────────┘
                         │
         ┌───────────────┴───────────────┐
         │                               │
┌────────▼────────┐            ┌────────▼────────┐
│  API Server 1   │            │  API Server N   │
│  Railway        │            │  Railway        │
│  - FastAPI      │            │  - FastAPI      │
│  - 2GB RAM      │            │  - 2GB RAM      │
│  - 2 vCPU       │            │  - 2 vCPU       │
└────────┬────────┘            └────────┬────────┘
         │                               │
         └───────────────┬───────────────┘
                         │
         ┌───────────────┴───────────────┐
         │                               │
┌────────▼────────┐            ┌────────▼────────┐
│  PostgreSQL     │            │  Redis          │
│  Railway        │            │  Railway        │
│  - 10GB storage │            │  - 1GB RAM      │
│  - Automated    │            │  - Persistence  │
│    backups      │            │    enabled      │
└─────────────────┘            └─────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   Background Worker                          │
│                   Railway (separate service)                 │
│                   - Daily data ingestion                     │
│                   - YOLO pattern detection                   │
│                   - Report generation                        │
│                   - Webhook delivery queue                   │
│                   - Backup service                           │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   External Storage                           │
│                   S3 / Azure Blob                            │
│                   - Database backups                         │
│                   - Audit log archives                       │
│                   - Generated reports                        │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   Monitoring Stack                           │
│                   - Prometheus (metrics)                     │
│                   - Grafana (dashboards)                     │
│                   - Loki (log aggregation)                   │
│                   - Sentry (error tracking)                  │
└─────────────────────────────────────────────────────────────┘
```

### Railway Configuration

**API Server Service** (`railway.toml`):

```toml
[build]
builder = "NIXPACKS"
buildCommand = """
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
  pip install -r requirements.txt && \
  pip install -e . && \
  pip install opencv-python-headless --force-reinstall --quiet
"""

[deploy]
startCommand = "uvicorn trader_koo.backend.main:app --host 0.0.0.0 --port $PORT --workers 4"
healthcheckPath = "/api/health"
healthcheckTimeout = 30
restartPolicyType = "ON_FAILURE"
restartPolicyMaxRetries = 3

[[deploy.environmentVariables]]
name = "ADMIN_STRICT_API_KEY"
value = "1"

[[deploy.environmentVariables]]
name = "DATABASE_URL"
value = "${{Postgres.DATABASE_URL}}"

[[deploy.environmentVariables]]
name = "REDIS_URL"
value = "${{Redis.REDIS_URL}}"
```

**Background Worker Service**:

```toml
[build]
builder = "NIXPACKS"
buildCommand = "pip install -r requirements.txt && pip install -e ."

[deploy]
startCommand = "python -m trader_koo.workers.main"
restartPolicyType = "ON_FAILURE"
restartPolicyMaxRetries = 10

[[deploy.environmentVariables]]
name = "WORKER_MODE"
value = "true"
```

### Health Checks

**Health check endpoint** (`/api/health`):

```python
@app.get("/api/health")
async def health_check():
    """
    Health check endpoint for load balancer
    Returns 200 if healthy, 503 if unhealthy
    """
    checks = {
        "database": await check_database_health(),
        "redis": await check_redis_health(),
        "disk_space": check_disk_space(),
        "memory": check_memory_usage(),
    }

    all_healthy = all(check["status"] == "healthy" for check in checks.values())

    return JSONResponse(
        status_code=200 if all_healthy else 503,
        content={
            "status": "healthy" if all_healthy else "unhealthy",
            "checks": checks,
            "timestamp": datetime.now().isoformat(),
            "version": VERSION,
        }
    )

async def check_database_health() -> dict:
    try:
        async with db.engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return {"status": "healthy", "latency_ms": 5}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

async def check_redis_health() -> dict:
    try:
        await redis.ping()
        return {"status": "healthy", "latency_ms": 2}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
```

### Zero-Downtime Deployment

**Blue-Green Deployment Strategy**:

1. Deploy new version to "green" environment
2. Run health checks on green environment
3. If healthy, route 10% of traffic to green
4. Monitor error rates and latency for 5 minutes
5. If metrics are good, gradually shift traffic (25%, 50%, 75%, 100%)
6. If metrics degrade, rollback to blue
7. Once 100% on green, keep blue running for 1 hour as backup
8. Decommission blue environment

**Database Migration Strategy**:

```python
# migrations/versions/001_add_users_table.py
from alembic import op
import sqlalchemy as sa

def upgrade():
    """
    Create users table
    This migration is backward compatible - old code can run without it
    """
    op.create_table(
        'users',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('username', sa.String(255), nullable=False),
        sa.Column('email', sa.String(255), nullable=False),
        sa.Column('password_hash', sa.String(255), nullable=False),
        sa.Column('role', sa.String(50), nullable=False),
        sa.Column('is_active', sa.Boolean(), default=True),
        sa.Column('created_at', sa.DateTime(), default=sa.func.now()),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('username'),
        sa.UniqueConstraint('email')
    )

    op.create_index('idx_users_email', 'users', ['email'])
    op.create_index('idx_users_username', 'users', ['username'])

def downgrade():
    """Rollback migration"""
    op.drop_table('users')
```

**Migration Execution**:

```bash
# Before deployment
alembic upgrade head --sql > migration.sql
# Review SQL, test on staging database
psql $STAGING_DATABASE_URL < migration.sql

# During deployment (automated)
alembic upgrade head
```

### Monitoring and Alerting

**Prometheus Metrics**:

```python
from prometheus_client import Counter, Histogram, Gauge

# Request metrics
http_requests_total = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

http_request_duration_seconds = Histogram(
    'http_request_duration_seconds',
    'HTTP request latency',
    ['method', 'endpoint']
)

# Database metrics
db_query_duration_seconds = Histogram(
    'db_query_duration_seconds',
    'Database query latency',
    ['query_type']
)

# LLM metrics
llm_requests_total = Counter(
    'llm_requests_total',
    'Total LLM requests',
    ['success']
)

llm_tokens_total = Counter(
    'llm_tokens_total',
    'Total LLM tokens used'
)

# System metrics
active_users = Gauge(
    'active_users',
    'Number of active users'
)

rate_limit_violations = Counter(
    'rate_limit_violations_total',
    'Total rate limit violations',
    ['endpoint']
)
```

**Grafana Dashboards**:

1. **System Overview**:

   - Request rate (req/s)
   - Error rate (%)
   - P50/P95/P99 latency
   - Active users
   - CPU/Memory usage

2. **Database Performance**:

   - Query latency by type
   - Connection pool usage
   - Slow query count
   - Database size

3. **External Services**:

   - LLM request rate and latency
   - Data source success rates
   - Webhook delivery success rate
   - Backup status

4. **Security**:
   - Failed authentication attempts
   - Rate limit violations
   - CORS rejections
   - Suspicious activity alerts

**Alert Rules** (Prometheus Alertmanager):

```yaml
groups:
  - name: platform_alerts
    rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }} (threshold: 0.05)"

      - alert: HighLatency
        expr: histogram_quantile(0.95, http_request_duration_seconds) > 0.5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High API latency detected"
          description: "P95 latency is {{ $value }}s (threshold: 0.5s)"

      - alert: DatabaseDown
        expr: up{job="postgres"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Database is down"

      - alert: HighRateLimitViolations
        expr: rate(rate_limit_violations_total[5m]) > 10
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High rate of rate limit violations"
          description: "{{ $value }} violations per second"
```

## Migration Strategy

### Phase 1: Security Hardening (Weeks 1-2)

**Objectives**: Implement all P0 security requirements without breaking existing functionality

**Tasks**:

1. **API Key Validation** (Req 1):

   - Add `ADMIN_STRICT_API_KEY` env var with default "1"
   - Add startup validation for API key presence and length
   - Add clear error messages with setup instructions
   - Test: Verify startup fails without key in strict mode

2. **LLM Output Sanitization** (Req 2, 7):

   - Define JSON schemas for all LLM response formats
   - Implement schema validation with jsonschema library
   - Add length limits and HTML/script stripping
   - Implement deterministic fallback for validation failures
   - Test: Property tests for all validation rules

3. **Email Token Expiration** (Req 3):

   - Add `expires_at` column to email tokens table
   - Set 7-day expiration on token creation
   - Add validation in authentication flow
   - Add admin endpoint for token revocation
   - Test: Property tests for expiration logic

4. **CORS Restrictive Defaults** (Req 4):

   - Default `TRADER_KOO_CORS_ORIGINS` to empty list
   - Add origin validation (https:// or http://localhost)
   - Log rejected CORS requests
   - Test: Property tests for origin validation

5. **Admin Auth Boundary Verification** (Req 5):

   - Create registry of admin endpoints
   - Add startup check for auth middleware
   - Fail startup if any admin endpoint lacks auth
   - Test: Unit tests for every admin endpoint

6. **Secret Exposure Hardening** (Req 6):

   - Create list of secret env var patterns
   - Implement log sanitization middleware
   - Sanitize error responses and stack traces
   - Audit /api/status, /api/health, /api/config endpoints
   - Test: Property tests for secret redaction

7. **Security Testing** (Req 47, 51):
   - Add CSRF protection middleware
   - Implement input sanitization for SQL injection
   - Implement input sanitization for XSS
   - Add request size and timeout limits
   - Implement bcrypt password hashing (12 rounds)
   - Add account lockout after 5 failed attempts
   - Test: Security test suite with penetration testing

**Migration Steps**:

- All changes are backward compatible
- No database schema changes required
- Deploy with feature flags to enable gradually
- Monitor error rates and rollback if needed

**Success Criteria**:

- All security tests pass
- Zero secrets in logs or API responses
- All admin endpoints protected
- No production incidents

### Phase 2: VIX Methodology Improvements (Weeks 3-4)

**Objectives**: Improve VIX analysis accuracy and reliability

**Tasks**:

1. **Trap/Reclaim Wording Clarity** (Req 8):

   - Update VIX engine to use consistent terminology
   - Add glossary to UI
   - Update all surfaces (API, UI, reports, emails)
   - Test: Property tests for pattern labeling

2. **Term Structure Fallback** (Req 9):

   - Implement VIX6M fallback when VIX3M unavailable
   - Implement synthetic term structure calculation
   - Add source labeling to all displays
   - Test: Unit tests for fallback scenarios

3. **Key Level Source Labeling** (Req 10):

   - Add source field to key level data structure
   - Update UI to display source labels
   - Update API responses to include source
   - Add source legend to VIX analysis tab
   - Test: Property tests for source labeling

4. **VIX Percentile Prominence** (Req 11):

   - Calculate percentile over 252-day window
   - Update UI to prominently display percentile
   - Add color coding (green/yellow/red)
   - Include in health score calculation
   - Test: Property tests for percentile calculation

5. **Multi-Source Data Redundancy** (Req 12):

   - Implement yfinance → Alpha Vantage → CSV fallback
   - Log data source for each fetch
   - Track success/failure rates
   - Add alerting for high failure rates
   - Test: Unit tests for fallback chain

6. **Adaptive Compression Thresholds** (Req 13):

   - Calculate dynamic thresholds based on 90-day percentile
   - Implement tight/moderate/wide threshold regimes
   - Display current thresholds in UI
   - Label signals with regime
   - Test: Property tests for threshold calculation

7. **Enhanced Trap/Reclaim Detection** (Req 14):
   - Implement bull/bear trap detection (3-bar reversal)
   - Implement support/resistance reclaim detection (2+ bars)
   - Calculate confidence based on volume and speed
   - Add visual markers to charts
   - Test: Property tests for pattern detection

**Migration Steps**:

- All changes are additive (no breaking changes)
- Deploy VIX engine updates independently
- A/B test new detection logic against old
- Gradually roll out to all users

**Success Criteria**:

- VIX analysis runs with multi-source redundancy
- Adaptive thresholds working correctly
- Trap/reclaim detection accurate on historical data
- No regression in existing VIX features

### Phase 3: Database Migration (Week 5)

**Objectives**: Migrate from SQLite to PostgreSQL

**Tasks**:

1. **Setup PostgreSQL**:

   - Provision PostgreSQL on Railway
   - Configure connection pooling (20 connections)
   - Set up read replica (optional)

2. **Schema Migration**:

   - Create all new tables (users, sessions, audit_logs, webhooks, etc.)
   - Add new columns to existing tables (data_source, detection_source)
   - Create indexes for performance
   - Set up partitioning for audit_logs

3. **Data Migration**:

   - Export data from SQLite
   - Transform data to match new schema
   - Import into PostgreSQL
   - Verify data integrity

4. **Dual-Write Period**:

   - Write to both SQLite and PostgreSQL
   - Read from SQLite (primary)
   - Compare results for consistency
   - Duration: 1 week

5. **Cutover**:
   - Switch reads to PostgreSQL
   - Monitor for issues
   - Keep SQLite as backup for 1 week
   - Remove SQLite code

**Migration Script**:

```python
# scripts/migrate_sqlite_to_postgres.py
import sqlite3
import psycopg2
from psycopg2.extras import execute_batch

def migrate_table(sqlite_conn, pg_conn, table_name, transform_fn=None):
    """Migrate a table from SQLite to PostgreSQL"""
    # Read from SQLite
    cursor = sqlite_conn.cursor()
    cursor.execute(f"SELECT * FROM {table_name}")
    rows = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]

    # Transform if needed
    if transform_fn:
        rows = [transform_fn(dict(zip(columns, row))) for row in rows]

    # Write to PostgreSQL
    pg_cursor = pg_conn.cursor()
    placeholders = ','.join(['%s'] * len(columns))
    query = f"INSERT INTO {table_name} ({','.join(columns)}) VALUES ({placeholders})"
    execute_batch(pg_cursor, query, rows)
    pg_conn.commit()

    print(f"Migrated {len(rows)} rows from {table_name}")

def main():
    sqlite_conn = sqlite3.connect("/data/trader_koo.db")
    pg_conn = psycopg2.connect(os.getenv("DATABASE_URL"))

    # Migrate tables
    migrate_table(sqlite_conn, pg_conn, "price_daily")
    migrate_table(sqlite_conn, pg_conn, "finviz_fundamentals")
    migrate_table(sqlite_conn, pg_conn, "options_iv")
    migrate_table(sqlite_conn, pg_conn, "yolo_patterns")
    migrate_table(sqlite_conn, pg_conn, "ingest_runs")
    migrate_table(sqlite_conn, pg_conn, "ingest_ticker_status")

    # Verify counts match
    verify_migration(sqlite_conn, pg_conn)

    sqlite_conn.close()
    pg_conn.close()

if __name__ == "__main__":
    main()
```

**Rollback Plan**:

- If issues detected, switch reads back to SQLite
- Fix issues in PostgreSQL
- Re-sync data from SQLite
- Retry cutover

### Phase 4: Core Enterprise Features (Weeks 6-8)

**Objectives**: Implement multi-user support, RBAC, audit logging, monitoring

**Tasks**:

1. **Redis Setup**:

   - Provision Redis on Railway
   - Configure for sessions, cache, rate limits
   - Set up persistence

2. **Authentication & RBAC** (Req 16):

   - Implement user registration and login
   - Implement JWT token generation
   - Implement role-based permissions
   - Add admin endpoints for user management
   - Test: Integration tests for auth flow

3. **Audit Logging** (Req 15):

   - Implement audit logger middleware
   - Log all admin actions and auth events
   - Implement query endpoint with filtering
   - Set up export to S3 for long-term retention
   - Test: Property tests for logging completeness

4. **Rate Limiting** (Req 17):

   - Implement Redis-backed rate limiter
   - Add middleware to enforce limits
   - Implement admin override capability
   - Test: Property tests for rate limiting

5. **Webhooks** (Req 18):

   - Implement webhook registration
   - Implement event triggering
   - Implement delivery with retry logic
   - Add HMAC signature support
   - Test: Integration tests for delivery

6. **Data Export** (Req 19):

   - Implement CSV/JSON/Excel export endpoints
   - Add date range filtering
   - Enforce rate limits on exports
   - Test: Unit tests for export formats

7. **Backup & Restore** (Req 20):

   - Implement automated backup service
   - Set up S3 storage
   - Implement restore endpoint
   - Add integrity verification
   - Test: Integration tests for backup/restore

8. **Health Monitoring** (Req 21):

   - Implement Prometheus metrics
   - Set up Grafana dashboards
   - Configure alerting rules
   - Test: Verify metrics are exported

9. **White-Label Support** (Req 22):
   - Implement branding configuration
   - Add logo upload endpoint
   - Add color scheme customization
   - Apply branding to all surfaces
   - Test: Unit tests for branding

**Migration Steps**:

- Deploy Redis first
- Deploy authentication (backward compatible with API key)
- Deploy other features incrementally
- Monitor each deployment for issues

**Success Criteria**:

- Multi-user system operational
- Audit trails complete
- Monitoring dashboards live
- 99.9% uptime achieved

### Phase 5: Competitive Features (Weeks 9-12) - P2

**High-Level Design Only**:

1. **Advanced Charting** (Req 23):

   - Integrate TradingView widget
   - Add custom indicator support
   - Persist user drawings

2. **Backtesting Engine** (Req 24):

   - Implement strategy definition DSL
   - Execute backtest over historical data
   - Calculate performance metrics
   - Generate equity curve

3. **Portfolio Tracking** (Req 25):

   - Add portfolio CRUD endpoints
   - Calculate P&L and metrics
   - Add alert thresholds

4. **Custom Screener** (Req 26):

   - Implement filter builder
   - Execute screen against ticker universe
   - Save and schedule screeners

5. **Custom Alerts** (Req 27):

   - Implement alert rule engine
   - Add multi-channel delivery (email, SMS, Slack)
   - Add cooldown to prevent spam

6. **Social Features** (Req 28):

   - Add trade idea posting
   - Implement commenting and voting
   - Calculate reputation scores

7. **Mobile API** (Req 29):

   - Add mobile-optimized endpoints
   - Implement push notification support
   - Add offline sync capability

8. **Compliance Reporting** (Req 30):
   - Generate regulatory reports
   - Add filtering and export
   - Implement 7-year retention

### Phase 6: Advanced Features (Weeks 13-16) - P3

**High-Level Design Only**:

1. **Earnings Calendar** (Req 31): Fetch and display earnings data
2. **Options Flow** (Req 32): Detect unusual options activity
3. **Sector Rotation** (Req 33): Track relative strength across sectors
4. **Correlation Matrix** (Req 34): Calculate and display correlations
5. **News Sentiment** (Req 35): Aggregate news with AI sentiment
6. **Economic Calendar** (Req 36): Display macro events
7. **Institutional Tracking** (Req 37): Track 13F filings and insider trades
8. **Pattern Scanner** (Req 38): Scan entire universe for patterns
9. **Risk Management Tools** (Req 39): Position sizing and risk calculators
10. **Real-Time WebSocket** (Req 40): Stream real-time price updates

### Rollback Procedures

**For each phase**:

1. **Immediate Rollback** (< 5 minutes):

   - Revert to previous Railway deployment
   - Use Railway's instant rollback feature
   - No data loss (database unchanged)

2. **Database Rollback** (< 30 minutes):

   - Run Alembic downgrade migration
   - Restore from most recent backup
   - Verify data integrity

3. **Partial Rollback** (feature flags):
   - Disable new features via environment variables
   - Keep infrastructure changes
   - Re-enable after fixes

**Rollback Triggers**:

- Error rate > 5% for 5 minutes
- P95 latency > 2 seconds for 5 minutes
- Critical security vulnerability discovered
- Data corruption detected
- User-reported critical bugs

### Testing Strategy Per Phase

**Phase 1 (Security)**:

- 100% test coverage for security code
- Penetration testing
- Security audit by external firm

**Phase 2 (VIX)**:

- Property tests for all detection logic
- Validation against historical VIX data
- A/B testing against old implementation

**Phase 3 (Database)**:

- Data integrity verification
- Performance benchmarking
- Load testing with production-like data

**Phase 4 (Enterprise)**:

- Integration tests for all features
- Load testing with 100 concurrent users
- Chaos engineering (kill services randomly)

**Phase 5-6 (Competitive)**:

- Feature-specific unit tests
- User acceptance testing
- Beta testing with select users

## Technology Stack

### Backend

- **Framework**: FastAPI (async Python web framework)
- **Language**: Python 3.11+
- **Database**: PostgreSQL 15 (migrating from SQLite)
- **Cache/Sessions**: Redis 7
- **ORM**: SQLAlchemy 2.0 with async support
- **Migrations**: Alembic
- **Task Queue**: APScheduler (in-process) → Celery (future)
- **Authentication**: JWT (PyJWT), bcrypt for password hashing
- **Validation**: Pydantic v2
- **Testing**: pytest, hypothesis (property-based testing)

### Frontend

- **Current**: Single-page vanilla JS + Plotly.js
- **Future**: Consider React/Vue for P2/P3 features if complexity warrants

### External Services

- **LLM**: Azure OpenAI (GPT-4)
- **Market Data**: yfinance (primary), Alpha Vantage (secondary)
- **Pattern Detection**: YOLOv8 (foduucom/stockmarket-pattern-detection-yolov8)
- **Email**: SMTP (Gmail) or Resend API
- **Storage**: S3 or Azure Blob Storage
- **Monitoring**: Prometheus + Grafana + Loki
- **Error Tracking**: Sentry

### Infrastructure

- **Hosting**: Railway (current), supports horizontal scaling
- **Load Balancer**: Railway built-in
- **CDN**: Cloudflare (optional for static assets)
- **CI/CD**: GitHub Actions

### Security

- **TLS**: Enforced in production (Railway handles termination)
- **Secrets Management**: Environment variables (Railway secrets)
- **Rate Limiting**: Redis-backed sliding window
- **Input Validation**: Pydantic models + custom sanitization
- **CSRF Protection**: FastAPI CSRF middleware
- **CORS**: Configurable whitelist

## API Endpoints

### Public Endpoints (No Auth Required)

- `GET /` - Serve frontend HTML
- `GET /api/health` - Health check for load balancer
- `GET /api/status` - System status (sanitized)

### Authenticated Endpoints (JWT Required)

- `GET /api/dashboard/{ticker}` - Dashboard data for ticker
- `GET /api/opportunities` - Trading opportunities
- `GET /api/vix/analysis` - VIX regime analysis
- `POST /api/analysis/comment` - Add comment (analyst/admin only)
- `GET /api/analysis/comments` - Get comments
- `GET /api/export/dashboard` - Export dashboard data (rate limited)
- `GET /api/export/patterns` - Export pattern history
- `GET /api/webhooks` - List user's webhooks
- `POST /api/webhooks` - Register webhook
- `DELETE /api/webhooks/{id}` - Delete webhook

### Admin Endpoints (X-API-Key or Admin JWT Required)

- `POST /api/admin/users` - Create user
- `PUT /api/admin/users/{id}` - Update user
- `DELETE /api/admin/users/{id}` - Deactivate user
- `GET /api/admin/audit-logs` - Query audit logs
- `GET /api/admin/run-yolo-seed` - Trigger YOLO detection
- `GET /api/admin/yolo-status` - YOLO pipeline status
- `POST /api/admin/trigger-update` - Trigger data update
- `GET /api/admin/daily-report` - Get latest report
- `POST /api/admin/email-latest-report` - Email report
- `POST /api/admin/backup` - Trigger manual backup
- `GET /api/admin/backups` - List backups
- `POST /api/admin/restore/{backup_id}` - Restore from backup
- `GET /api/admin/metrics` - Prometheus metrics
- `GET /api/admin/rate-limits` - View rate limit status
- `POST /api/admin/rate-limits/{user_id}` - Override rate limit

### Authentication Endpoints

- `POST /api/auth/register` - Register new user (if enabled)
- `POST /api/auth/login` - Login with username/password
- `POST /api/auth/refresh` - Refresh JWT token
- `POST /api/auth/logout` - Invalidate session
- `POST /api/auth/request-email-token` - Request email login link
- `GET /api/auth/verify-email-token` - Verify email token
- `POST /api/auth/change-password` - Change password

## Data Flow Diagrams

### Dashboard Request Flow

```
User → Load Balancer → API Server
                          ↓
                    Check JWT/Auth
                          ↓
                    Check Rate Limit (Redis)
                          ↓
                    Check Cache (Redis)
                          ↓ (cache miss)
                    Query Database (PostgreSQL)
                          ↓
                    Fetch Market Data (yfinance/fallback)
                          ↓
                    Run Pattern Detection
                          ↓
                    Query YOLO Patterns (PostgreSQL)
                          ↓
                    Merge Results
                          ↓
                    Cache Result (Redis, 5min TTL)
                          ↓
                    Log to Audit (PostgreSQL)
                          ↓
                    Return JSON Response
```

### Daily Update Pipeline

```
Scheduler (22:00 UTC) → Background Worker
                              ↓
                        Fetch S&P 500 Ticker List
                              ↓
                        For Each Ticker:
                              ↓
                        Try yfinance → Alpha Vantage → CSV
                              ↓
                        Store in PostgreSQL
                              ↓
                        Log Ingest Status
                              ↓
                        Run YOLO Detection (daily + weekly)
                              ↓
                        Store Patterns in PostgreSQL
                              ↓
                        Generate Daily Report
                              ↓
                        Trigger Webhooks (report_generated)
                              ↓
                        Email Report (if configured)
                              ↓
                        Create Backup
                              ↓
                        Upload to S3
```

### Webhook Delivery Flow

```
Event Occurs → Webhook Service
                    ↓
              Query Registered Webhooks (PostgreSQL)
                    ↓
              For Each Webhook:
                    ↓
              Build JSON Payload
                    ↓
              Add HMAC Signature (if configured)
                    ↓
              POST to Webhook URL (10s timeout)
                    ↓
              Success? → Log Delivery (PostgreSQL)
                    ↓
              Failure? → Retry with Backoff (1s, 2s, 4s)
                    ↓
              After 3 Failures → Log Final Failure
```

## Security Architecture

### Defense in Depth Layers

1. **Network Layer**:

   - TLS 1.2+ enforced
   - DDoS protection (Cloudflare)
   - IP-based rate limiting

2. **Application Layer**:

   - Authentication (JWT, API keys)
   - Authorization (RBAC)
   - Rate limiting (per-user, per-IP)
   - Request size limits (10MB)
   - Request timeout limits (30s)

3. **Input Validation Layer**:

   - Pydantic schema validation
   - SQL injection prevention (parameterized queries)
   - XSS prevention (input sanitization)
   - CSRF protection

4. **Output Sanitization Layer**:

   - Secret redaction in logs
   - Secret redaction in error responses
   - HTML escaping for user content
   - LLM output validation and sanitization

5. **Data Layer**:

   - Encrypted at rest (database encryption)
   - Encrypted in transit (TLS)
   - Password hashing (bcrypt, 12 rounds)
   - Audit logging (immutable)

6. **Monitoring Layer**:
   - Security event logging
   - Failed auth attempt tracking
   - Rate limit violation tracking
   - Anomaly detection alerts

### Threat Mitigation Summary

| Threat                    | Mitigation                                       | Requirements |
| ------------------------- | ------------------------------------------------ | ------------ |
| Unauthorized admin access | API key validation, JWT auth, RBAC               | 1, 5, 16, 47 |
| Secret exposure           | Redaction in logs/errors, sanitized endpoints    | 6, 47        |
| LLM injection             | Schema validation, output sanitization, fallback | 2, 7         |
| SQL injection             | Parameterized queries, input validation          | 47           |
| XSS attacks               | Input sanitization, output escaping, CSP headers | 47           |
| CSRF attacks              | CSRF tokens, SameSite cookies                    | 47           |
| DoS attacks               | Rate limiting, request size/timeout limits       | 17, 47       |
| Session hijacking         | Secure cookies, session expiration, HTTPS        | 3, 47        |
| Data breach               | Encryption at rest/transit, audit logging        | 47, 48       |

## Summary

This design document specifies the comprehensive upgrade of trader_koo from a personal tool to an enterprise-grade platform. The design prioritizes:

1. **Security first**: Defense-in-depth with multiple layers of protection, zero-trust architecture, comprehensive audit logging
2. **Reliability**: Multi-source data redundancy, graceful degradation, circuit breakers, 99.9% uptime target
3. **Scalability**: Stateless API servers, Redis for shared state, PostgreSQL with connection pooling, horizontal scaling ready
4. **Observability**: Prometheus metrics, Grafana dashboards, structured logging, distributed tracing
5. **Correctness**: 49 testable properties with property-based testing, 70%+ code coverage, comprehensive security testing

The migration follows a phased approach over 16 weeks, with P0 security and VIX improvements in weeks 1-4, core enterprise features in weeks 5-8, and competitive features in weeks 9-16. Each phase includes detailed rollback procedures and success criteria.

The architecture supports the current single-instance deployment on Railway while enabling future horizontal scaling. All external dependencies have fallback mechanisms to ensure system reliability even when third-party services fail.
