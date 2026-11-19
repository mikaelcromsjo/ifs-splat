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

## 2. Stop the splat demo being served from cache

**Goal.** `https://mikaelcromsjo.com/splat/` is republished several times a day, and nginx sends
no `Cache-Control` for it — only `Last-Modified` and `ETag`. A browser is then free to apply
heuristic freshness and serve the copy it already has, and an ES module already in the module map
is not revalidated on a soft reload at all, so the page comes up part new and part old. The
published copy now stamps every module URL with the build id, which handles the modules without
root; `index.html` itself is the one file that cannot be stamped by the thing it names, so it
needs the header.

**Step 1.** Edit `/etc/nginx/sites-available/mikaelcromsjo`. If it does not already contain the
marker `# ifs-splat-nocache`, insert this block immediately before `location / {`:

```nginx
    # ifs-splat-nocache — the demo is republished often; never serve a cached page
    location /splat/ {
        add_header Cache-Control "no-store" always;
    }
```

Safe to run twice: the marker check makes a second run a no-op. A prefix location, so the
`\.(ply|splat)$` regex block from task 1 (if it is in place) still wins for `tree.ply` and keeps
its own caching — regex locations are matched after prefix ones and take precedence.

**Step 2.** `nginx -t`

**Step 3.** `systemctl reload nginx`

**Check.**

    curl -sI https://mikaelcromsjo.com/splat/ | grep -i 'cache-control'
    curl -sI https://mikaelcromsjo.com/splat/js/viewer.js | grep -i 'cache-control'

Expect `Cache-Control: no-store` from both. One header, not two — if `expires` is set anywhere
for this path it will add a second one, which is the thing to watch for.

**Rollback.** Remove the block, `nginx -t`, `systemctl reload nginx`.
