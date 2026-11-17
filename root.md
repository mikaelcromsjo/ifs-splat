# Root tasks — ifs-splat

## 1. Serve .ply/.splat with CORS so web viewers can load them

**Goal.** `https://mikaelcromsjo.com/splat/tree.ply` is already public and returns 200, but it
carries no `Access-Control-Allow-Origin`. Browser splat viewers hosted elsewhere (SuperSplat,
antimatter15/splat) fetch cross-origin, so they currently fail. Add the header for these two
extensions only, and give them an explicit content type.

**Step 1.** Edit `/etc/nginx/sites-available/mikaelcromsjo`. If it does not already contain the
marker `# ifs-splat`, insert this block immediately before `location / {`:

```nginx
    # ifs-splat — gaussian splat data, fetched cross-origin by web viewers
    location ~ \.(ply|splat)$ {
        default_type application/octet-stream;
        add_header Access-Control-Allow-Origin "*" always;
        add_header Cache-Control "public, max-age=3600" always;
    }
```

Safe to run twice: the marker check makes a second run a no-op.

**Step 2.** `nginx -t`

**Step 3.** `systemctl reload nginx`

**Check.**

    curl -sI https://mikaelcromsjo.com/splat/tree.ply | grep -i 'access-control\|content-type'

Expect `Access-Control-Allow-Origin: *` and `Content-Type: application/octet-stream`.

**Rollback.** Remove the block, `nginx -t`, `systemctl reload nginx`.
