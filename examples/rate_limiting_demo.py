"""Demo script for rate limiting functionality.

This script demonstrates:
1. Basic rate limiting enforcement
2. Sliding window algorithm
3. Admin overrides
4. Status monitoring
"""

import time
from datetime import timedelta

from trader_koo.ratelimit.service import RateLimiter, RateLimitConfig


def demo_basic_rate_limiting():
    """Demonstrate basic rate limiting."""
    print("\n" + "="*60)
    print("DEMO 1: Basic Rate Limiting")
    print("="*60)
    
    # Create rate limiter with low limit for demo
    config = RateLimitConfig(
        public_limit=5,
        public_window=timedelta(seconds=10),
    )
    limiter = RateLimiter(config)
    
    print(f"\nConfiguration: {config.public_limit} requests per {config.public_window}")
    print("\nMaking requests...")
    
    key = "demo_user"
    
    # Make requests up to the limit
    for i in range(7):
        result = limiter.check_rate_limit(key, config.public_limit, config.public_window)
        
        if result.allowed:
            print(f"  Request {i+1}: ✓ ALLOWED (remaining: {result.remaining})")
        else:
            print(f"  Request {i+1}: ✗ DENIED (retry after: {result.retry_after}s)")


def demo_sliding_window():
    """Demonstrate sliding window algorithm."""
    print("\n" + "="*60)
    print("DEMO 2: Sliding Window Algorithm")
    print("="*60)
    
    config = RateLimitConfig(
        public_limit=3,
        public_window=timedelta(seconds=5),
    )
    limiter = RateLimiter(config)
    
    print(f"\nConfiguration: {config.public_limit} requests per {config.public_window}")
    print("\nPhase 1: Use up the limit")
    
    key = "demo_user"
    
    # Use up the limit
    for i in range(3):
        result = limiter.check_rate_limit(key, config.public_limit, config.public_window)
        print(f"  Request {i+1}: {'✓ ALLOWED' if result.allowed else '✗ DENIED'}")
    
    # Try one more (should be denied)
    result = limiter.check_rate_limit(key, config.public_limit, config.public_window)
    print(f"  Request 4: {'✓ ALLOWED' if result.allowed else '✗ DENIED'}")
    
    print("\nPhase 2: Wait for window to slide (6 seconds)...")
    time.sleep(6)
    
    print("Phase 3: Try again after window expires")
    result = limiter.check_rate_limit(key, config.public_limit, config.public_window)
    print(f"  Request 5: {'✓ ALLOWED' if result.allowed else '✗ DENIED'} (window has slid)")


def demo_admin_override():
    """Demonstrate admin override functionality."""
    print("\n" + "="*60)
    print("DEMO 3: Admin Override")
    print("="*60)
    
    config = RateLimitConfig(
        public_limit=3,
        public_window=timedelta(seconds=10),
    )
    limiter = RateLimiter(config)
    
    print(f"\nConfiguration: {config.public_limit} requests per {config.public_window}")
    
    key = "vip_user"
    
    print("\nPhase 1: Use up normal limit")
    for i in range(3):
        result = limiter.check_rate_limit(key, config.public_limit, config.public_window)
        print(f"  Request {i+1}: {'✓ ALLOWED' if result.allowed else '✗ DENIED'}")
    
    # Should be denied
    result = limiter.check_rate_limit(key, config.public_limit, config.public_window)
    print(f"  Request 4: {'✓ ALLOWED' if result.allowed else '✗ DENIED'}")
    
    print("\nPhase 2: Admin sets override (limit: 100, duration: 60s)")
    limiter.set_override(
        key=key,
        limit=100,
        window=timedelta(seconds=10),
        duration=timedelta(seconds=60),
    )
    
    print("Phase 3: Try again with override")
    result = limiter.check_rate_limit(key, config.public_limit, config.public_window)
    print(f"  Request 5: {'✓ ALLOWED' if result.allowed else '✗ DENIED'} (override active)")
    
    # Check status
    status = limiter.get_status(key)
    print(f"\nStatus: {status['request_count']} requests, override active: {status['has_override']}")


def demo_status_monitoring():
    """Demonstrate status monitoring."""
    print("\n" + "="*60)
    print("DEMO 4: Status Monitoring")
    print("="*60)
    
    config = RateLimitConfig(
        public_limit=10,
        public_window=timedelta(seconds=60),
    )
    limiter = RateLimiter(config)
    
    print("\nMaking requests from multiple users...")
    
    # Simulate requests from different users
    users = ["user1", "user2", "user3"]
    for user in users:
        for _ in range(3):
            limiter.check_rate_limit(user, config.public_limit, config.public_window)
        print(f"  {user}: 3 requests made")
    
    print("\nGetting status for all users:")
    all_status = limiter.get_all_status()
    
    for status in all_status:
        print(f"\n  Key: {status['key']}")
        print(f"    Request count: {status['request_count']}")
        print(f"    Oldest request: {status['oldest_request']}")
        print(f"    Has override: {status['has_override']}")


def demo_per_ip_vs_per_user():
    """Demonstrate per-IP vs per-user rate limiting."""
    print("\n" + "="*60)
    print("DEMO 5: Per-IP vs Per-User Rate Limiting")
    print("="*60)
    
    config = RateLimitConfig(
        public_limit=5,
        public_window=timedelta(minutes=1),
        authenticated_limit=20,
        authenticated_window=timedelta(minutes=1),
    )
    limiter = RateLimiter(config)
    
    print("\nScenario 1: Public endpoint (per-IP)")
    print(f"  Limit: {config.public_limit} requests per minute")
    
    ip1 = "ip:192.168.1.1"
    ip2 = "ip:192.168.1.2"
    
    # Use up limit for IP1
    for _ in range(5):
        limiter.check_rate_limit(ip1, config.public_limit, config.public_window)
    
    result1 = limiter.check_rate_limit(ip1, config.public_limit, config.public_window)
    result2 = limiter.check_rate_limit(ip2, config.public_limit, config.public_window)
    
    print(f"  IP1 (192.168.1.1): {'✓ ALLOWED' if result1.allowed else '✗ DENIED'} (used up limit)")
    print(f"  IP2 (192.168.1.2): {'✓ ALLOWED' if result2.allowed else '✗ DENIED'} (independent limit)")
    
    print("\nScenario 2: Authenticated endpoint (per-user)")
    print(f"  Limit: {config.authenticated_limit} requests per minute")
    
    user1 = "user:alice"
    user2 = "user:bob"
    
    # Make some requests
    for _ in range(10):
        limiter.check_rate_limit(user1, config.authenticated_limit, config.authenticated_window)
    
    result1 = limiter.check_rate_limit(user1, config.authenticated_limit, config.authenticated_window)
    result2 = limiter.check_rate_limit(user2, config.authenticated_limit, config.authenticated_window)
    
    print(f"  User alice: {'✓ ALLOWED' if result1.allowed else '✗ DENIED'} (remaining: {result1.remaining})")
    print(f"  User bob: {'✓ ALLOWED' if result2.allowed else '✗ DENIED'} (remaining: {result2.remaining})")


def main():
    """Run all demos."""
    print("\n" + "="*60)
    print("RATE LIMITING DEMO")
    print("="*60)
    print("\nThis demo shows the rate limiting functionality in action.")
    print("Each demo illustrates a different aspect of the system.")
    
    demo_basic_rate_limiting()
    demo_sliding_window()
    demo_admin_override()
    demo_status_monitoring()
    demo_per_ip_vs_per_user()
    
    print("\n" + "="*60)
    print("DEMO COMPLETE")
    print("="*60)
    print("\nFor more information, see trader_koo/ratelimit/README.md")


if __name__ == "__main__":
    main()
