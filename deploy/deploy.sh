#!/usr/bin/env bash
# Wrangler-only deploy: static site + functions/_middleware.js (Basic Auth).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUNDLE="$ROOT/deploy/drag-trigger"
PROJECT_NAME="${PAGES_PROJECT_NAME:-drag-trigger}"
WRANGLER=(npx --yes wrangler@latest)

"$ROOT/deploy/sync-website.sh"
cd "$BUNDLE"

if [[ ! -f .dev.vars ]]; then
  cp .dev.vars.example .dev.vars
  echo ""
  echo "Created deploy/drag-trigger/.dev.vars"
  echo "  → Edit BASIC_AUTH_PASS, then run:  ./deploy/deploy.sh"
  echo ""
  exit 1
fi

# shellcheck disable=SC1091
set -a && source .dev.vars && set +a

BASIC_AUTH_USER="${BASIC_AUTH_USER:-reviewer}"

if [[ -z "${BASIC_AUTH_PASS:-}" ]]; then
  echo "Error: BASIC_AUTH_PASS is empty in .dev.vars"
  exit 1
fi

echo "Checking Wrangler login..."
"${WRANGLER[@]}" whoami >/dev/null 2>&1 || {
  echo "Run once:  npx wrangler login"
  exit 1
}

echo "Ensuring Pages project exists: $PROJECT_NAME"
"${WRANGLER[@]}" pages project create "$PROJECT_NAME" --production-branch=main 2>/dev/null || true

echo "Uploading auth secrets to Cloudflare..."
printf '%s' "$BASIC_AUTH_USER" | "${WRANGLER[@]}" pages secret put BASIC_AUTH_USER \
  --project-name="$PROJECT_NAME"
printf '%s' "$BASIC_AUTH_PASS" | "${WRANGLER[@]}" pages secret put BASIC_AUTH_PASS \
  --project-name="$PROJECT_NAME"

echo "Deploying (static assets + functions/_middleware.js)..."
"${WRANGLER[@]}" pages deploy . --project-name="$PROJECT_NAME" --commit-dirty=true

echo ""
echo "Done. Open your *.pages.dev URL or dragtrigger.snsnegi.com"
echo "  Username: $BASIC_AUTH_USER"
echo "  Password: (from .dev.vars)"
