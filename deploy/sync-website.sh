#!/usr/bin/env bash
# Copy the latest site from _exploration/websiteV2 into the Cloudflare upload bundle.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/_exploration/websiteV2"
DST="$ROOT/deploy/drag-trigger"

cp "$SRC/index.html" "$SRC/styles.css" "$SRC/script.js" "$DST/"
echo "Synced website assets → deploy/drag-trigger/"
