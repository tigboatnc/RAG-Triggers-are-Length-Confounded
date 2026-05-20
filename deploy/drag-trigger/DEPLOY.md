# Deploy with Wrangler only

Dashboard drag-and-drop **does not** run `functions/`. Use **`./deploy/deploy.sh`** instead.

## One-time setup

```bash
npx wrangler login
cp deploy/drag-trigger/.dev.vars.example deploy/drag-trigger/.dev.vars
# edit .dev.vars → set BASIC_AUTH_PASS
```

Attach custom domain in Cloudflare dashboard: **dragtrigger.snsnegi.com** → this Pages project.

## Every deploy

```bash
./deploy/sync-website.sh   # optional if you edited websiteV2
./deploy/deploy.sh
```

Login in the browser:

- **Username:** `reviewer` (in `wrangler.toml`)
- **Password:** whatever you put in `.dev.vars`

## Change password

Edit `.dev.vars`, run `./deploy/deploy.sh` again.

## Go public

Delete `functions/_middleware.js`, run `./deploy/deploy.sh`.
