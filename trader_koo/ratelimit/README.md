# Rate Limiting Module

Comprehensive API rate limiting implementation for trader_koo platform.

## Features

- **Per-IP rate limiting** for public endpoints (100 requests/minute)
- **Per-user rate limiting** for authenticated endpoints (1000 requests/hour)
- **Sliding window algorithm** for accurate rate calculation
- **HTTP 429 responses** with Retry-After headers
- **Admin override capabilities** to temporarily increase limits
- **Admin status endpoints** to monitor rate limit usage
- **Violation logging** for security monitoring
- **In-memory storage** with Redis support ready

## Requirements Implemented

- **Requirement 17.1**: Per-IP limits (100/min) and per-user limits (1000/hour)
- **Requirement 17.2**: HTTP 429 with Retry-After header when exceeded
- **Requirement 17.3**: Sliding window algorithm for accurate rate calculation
- **Requirement 17.4**: Sliding window implementation
- **Requirement 17.5**: State storage (in-memory, Redis-ready)
- **Requirement 17.6**: Admin status endpoint to view current limits
- **Requirement 17.7**: Admin overrides to temporarily increase limits
- **Requirement 17.8**: Violation logging with IP, user, endpoint

## Architecture

### Components

1. **RateLimiter** (`service.py`): Core rate limiting logic with sliding window algorithm
2. **RateLimitMiddleware** (`middleware.py`): FastAPI middleware for request interception
3. **Admin API** (`api.py`): Endpoints for monitoring and management
4. **Integration** (`integration.py`): Setup and configuration helper

### Data Flow

```
Request → Middleware → RateLimiter.check_rate_limit()
                    ↓
                 Allowed?
                ↙       ↘
              Yes        No
               ↓          ↓
         Next Handler   HTTP 429
               ↓          ↓
         Add Headers   Retry-After
               ↓          ↓
           Response    Log Violation
```

## Configuration

Set environment variables to customize rate limits:

```bash
# Per-IP limits for public endpoints
RATE_LIMIT_PUBLIC_PER_MINUTE=100

# Per-user limits for authenticated endpoints
RATE_LIMIT_AUTH_PER_HOUR=1000

# Export endpoint limits
RATE_LIMIT_EXPORT_PER_HOUR=10
```

## Usage

### Basic Setup

The rate limiting is automatically initialized in `main.py`:

```python
from trader_koo.ratelimit.integration import initialize_rate_limiting

# In lifespan function
rate_limiter = initialize_rate_limiting(app)
```

### Admin API Endpoints

#### Get All Rate Limit Status

```bash
GET /api/admin/ratelimit/status
```

Returns rate limit status for all tracked keys.

#### Get Status for Specific Key

```bash
GET /api/admin/ratelimit/status/{key}
```

Example: `/api/admin/ratelimit/status/ip:192.168.1.1`

#### Set Rate Limit Override

```bash
POST /api/admin/ratelimit/override
Content-Type: application/json

{
  "key": "user:john_doe",
  "limit": 5000,
  "window_seconds": 3600,
  "duration_seconds": 86400
}
```

Temporarily increases rate limit for a specific user or IP.

#### Remove Rate Limit Override

```bash
DELETE /api/admin/ratelimit/override/{key}
```

#### Cleanup Expired Entries

```bash
POST /api/admin/ratelimit/cleanup?max_age_hours=24
```

## Rate Limit Response Headers

All successful responses include rate limit headers:

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 2024-01-15T10:31:00Z
```

When rate limit is exceeded (HTTP 429):

```
Retry-After: 45
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 2024-01-15T10:31:00Z
```

## Sliding Window Algorithm

The rate limiter uses a sliding window algorithm for accurate rate calculation:

1. Each request timestamp is stored
2. On each check, timestamps outside the window are removed
3. Current count is compared against the limit
4. If allowed, current timestamp is added

This provides more accurate rate limiting than fixed windows, preventing burst traffic at window boundaries.

## Logging

Rate limit violations are logged with full context:

```
WARNING: Rate limit exceeded: key=ip:192.168.1.1, ip=192.168.1.1,
         user=None, endpoint=/api/data, method=GET
```

## Testing

Run unit tests:

```bash
pytest tests/test_rate_limiting.py -v
```

Run integration tests:

```bash
pytest tests/test_rate_limiting_integration.py -v
```

## Future Enhancements

### Redis Backend

For production deployments with multiple instances, implement Redis backend:

```python
class RedisRateLimiter(RateLimiter):
    def __init__(self, redis_client, config):
        self.redis = redis_client
        self.config = config

    def check_rate_limit(self, key, limit, window):
        # Use Redis sorted sets for distributed rate limiting
        # ZADD key timestamp timestamp
        # ZREMRANGEBYSCORE key -inf (now - window)
        # ZCARD key
        pass
```

### Rate Limit Tiers

Add configurable rate limit tiers:

```python
RATE_LIMIT_TIERS = {
    "free": (100, timedelta(hours=1)),
    "pro": (1000, timedelta(hours=1)),
    "enterprise": (10000, timedelta(hours=1)),
}
```

### Burst Allowance

Allow short bursts above the limit:

```python
config = RateLimitConfig(
    limit=100,
    window=timedelta(minutes=1),
    burst_allowance=20,  # Allow 120 requests in short burst
)
```

## Security Considerations

1. **IP Spoofing**: The middleware checks `X-Forwarded-For` header. Ensure your load balancer/proxy sets this correctly.

2. **DDoS Protection**: Rate limiting helps but is not a complete DDoS solution. Use additional layers (CloudFlare, AWS Shield, etc.).

3. **Admin Overrides**: Only accessible via admin endpoints with authentication. Log all override operations.

4. **Memory Usage**: In-memory storage grows with unique IPs/users. The cleanup endpoint helps manage this.

## Troubleshooting

### Rate Limit Not Working

1. Check middleware is added: `app.add_middleware(RateLimitMiddleware)`
2. Verify configuration: Check environment variables
3. Check logs: Look for rate limiter initialization messages

### False Positives

1. Check if multiple users share an IP (NAT, corporate proxy)
2. Consider using per-user limits instead of per-IP
3. Set admin override for affected IPs

### High Memory Usage

1. Run cleanup endpoint regularly: `/api/admin/ratelimit/cleanup`
2. Reduce `max_age_hours` parameter
3. Consider implementing Redis backend for distributed storage

## References

- Design Document: `.kiro/specs/enterprise-platform-upgrade/design.md`
- Requirements: `.kiro/specs/enterprise-platform-upgrade/requirements.md` (Requirement 17)
- FastAPI Middleware: https://fastapi.tiangolo.com/advanced/middleware/
- Rate Limiting Algorithms: https://en.wikipedia.org/wiki/Rate_limiting
