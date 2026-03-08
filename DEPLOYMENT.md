# Deployment Guide

## Best Practices

### 1. Use Staging Environment First

**Railway Setup:**

- Create two services: `trader-koo-staging` and `trader-koo-production`
- Both use the same codebase but different branches/environments
- Test all changes in staging before promoting to production

**Branch Strategy:**

```
main (production) ← merge from staging after testing
staging (staging environment) ← merge from feature branches
feature/* (development)
```

### 2. Pre-Deployment Checklist

Before pushing to production:

- [ ] All tests pass locally
- [ ] Code reviewed and approved
- [ ] Tested in staging environment
- [ ] Database migrations tested (if any)
- [ ] Environment variables verified
- [ ] No secrets in code
- [ ] Diagnostics clean (`getDiagnostics`)

### 3. Deployment Process

#### Option A: Zero-Downtime (Recommended)

1. **Deploy to staging first:**

   ```bash
   git checkout staging
   git merge feature/your-feature
   git push origin staging
   ```

2. **Test staging thoroughly:**

   - Check `/api/health`
   - Verify all admin endpoints work
   - Test critical user flows
   - Monitor logs for errors

3. **Promote to production:**
   ```bash
   git checkout main
   git merge staging
   git push origin main
   ```

#### Option B: Maintenance Mode (For Major Changes)

1. **Enable maintenance page:**

   - Temporarily modify `trader_koo/backend/main.py` to serve maintenance.html
   - Or use Railway's built-in maintenance mode

2. **Deploy changes:**

   ```bash
   git push origin main
   ```

3. **Verify deployment:**

   - Check Railway logs
   - Test `/api/health`
   - Verify database integrity

4. **Disable maintenance mode**

### 4. Rollback Strategy

If deployment fails:

**Quick Rollback (Railway):**

1. Go to Railway dashboard → Deployments
2. Find the last working deployment
3. Click "Redeploy"

**Git Rollback:**

```bash
git revert HEAD
git push origin main
```

### 5. Monitoring After Deployment

Check these within 5 minutes of deployment:

- [ ] `/api/health` returns 200
- [ ] `/api/status` shows correct data
- [ ] Railway logs show no errors
- [ ] Database has expected record count
- [ ] Admin endpoints require authentication

## Railway Environment Setup

### Staging Environment

```bash
# Railway CLI
railway link trader-koo-staging

# Set environment variables
railway variables set TRADER_KOO_DB_PATH=/data/trader_koo_staging.db
railway variables set TRADER_KOO_ALLOWED_ORIGIN=https://trader-koo-staging.up.railway.app
railway variables set TRADER_KOO_APP_URL=https://trader-koo-staging.up.railway.app
```

### Production Environment

```bash
# Railway CLI
railway link trader-koo-production

# Set environment variables (use production values)
railway variables set TRADER_KOO_DB_PATH=/data/trader_koo.db
railway variables set TRADER_KOO_ALLOWED_ORIGIN=https://trader.kooexperience.com
railway variables set TRADER_KOO_APP_URL=https://trader.kooexperience.com
```

## Emergency Procedures

### Site is Down

1. **Check Railway status:**

   - Go to Railway dashboard
   - Check deployment logs
   - Look for error messages

2. **Quick fixes:**

   - Redeploy last working version
   - Check environment variables
   - Verify volume is mounted

3. **Enable maintenance page:**
   - Serve `maintenance.html` from root route
   - Investigate issue without user-facing errors

### Database Issues

1. **Check database stats:**

   ```bash
   curl -H "X-API-Key: YOUR_KEY" \
     https://trader.kooexperience.com/api/admin/database-stats
   ```

2. **Backup before fixes:**

   - Railway volumes are persistent
   - Download backup via Railway CLI if needed

3. **Recovery:**
   - Restore from backup
   - Re-run data seed if necessary

## Current Deployment Issue (2026-03-08)

**Problem:** 25 admin endpoints missing `@require_admin_auth` decorator

**Fix Applied:**

- Added `require_admin_auth` import to main.py
- Added decorator to all 25 admin endpoints
- Ready to commit and push

**Next Steps:**

1. Commit the fix
2. Push to production
3. Monitor deployment logs
4. Verify all endpoints work

## Maintenance Page Usage

To enable maintenance mode manually:

1. **Modify main.py temporarily:**

   ```python
   @app.get("/")
   def root():
       return FileResponse(FRONTEND_INDEX.parent / "maintenance.html")
   ```

2. **Push to production**

3. **Fix the issue**

4. **Revert maintenance mode and push again**

## Tips

- Always test in staging first
- Keep deployments small and frequent
- Monitor logs during deployment
- Have rollback plan ready
- Document any manual steps needed
- Use Railway's deployment history for quick rollbacks
