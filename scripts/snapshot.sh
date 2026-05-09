#!/usr/bin/env bash
# Phase snapshot — package the repo state into a single tar.gz that survives
# leaving this rented compute box. Includes everything worth saving:
#
#   - All markdown docs and TeX (_PLAN.md, _AGENT.md, _findings.md, _draft.tex,
#     README.md, draft.md, extras/)
#   - All scripts and tools (scripts/, tools/)
#   - Environment manifest (environment.yml, .gitignore)
#   - All experiment outputs in runs/:
#       *.jsonl, *.log, *.json, *.txt — the per-step features, summaries, logs
#       *.npz                          — heavy CRV vector dumps (see policy below)
#   - All figures (figs/*.png — the plots from each phase)
#   - Full git history (.git/) — committed state replays cleanly on the
#     destination, and any uncommitted working-tree changes show up via
#     `git status` after extraction
#
# Excludes:
#   - models/  (~180 GB at Phase 2; re-pull via scripts/download_models.sh, SHAs pinned)
#   - data/    (datasets; re-pull from source)
#   - runs_setup*.log  (verbose pip install logs; not science)
#   - .venv/, __pycache__/, conda env caches
#   - the previous boot tarball (recollection-patch.tar.gz) — already in git
#     history; skip the working-tree copy to avoid recursive nesting
#
# .npz inclusion policy (runs/**/*.npz)
#   At Phase 2 the per-trace CRV .npz dumps total ~55 MB across 18 files
#   (3 models × 3 prompts × 2 layer modes). We INCLUDE them in snapshots
#   because (a) regenerating them costs ~90 minutes of GPU time, (b) future
#   analysis (PCA, clustering, separability tests) wants the raw vectors, and
#   (c) at <100 MB they don't bloat the tarball. They are still gitignored
#   (heavy binary blobs don't belong in git history); snapshots are the
#   correct vehicle for them.
#
#   When future phases (Phase 4 showdown) generate larger traces and runs/
#   approaches 1 GB, switch to `--no-heavy` to exclude the .npz files. The
#   smaller "the science" set (jsonl + log + json + txt) is always included
#   and stays under 5 MB even through Phase 4.
#
# Usage:
#   bash scripts/snapshot.sh                       # tag = "snapshot"
#   bash scripts/snapshot.sh phase1                # tag = "phase1"
#   bash scripts/snapshot.sh phase2 /custom/dir    # phase tag + custom output dir
#   bash scripts/snapshot.sh phase4 "" --no-heavy  # exclude runs/**/*.npz

set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PHASE_TAG="${1:-snapshot}"
OUT_DIR="${2:-/workspace/snapshots}"
NO_HEAVY="${3:-}"            # pass "--no-heavy" to skip runs/**/*.npz
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
REPO_NAME="$(basename "$PROJECT_ROOT")"
HEAVY_TAG=""
if [[ "$NO_HEAVY" == "--no-heavy" ]]; then
  HEAVY_TAG="-noheavy"
fi
ARCHIVE="${OUT_DIR}/${REPO_NAME}-${PHASE_TAG}${HEAVY_TAG}-${STAMP}.tar.gz"

green() { printf "\033[32m[ok]\033[0m %s\n" "$1"; }
yellow() { printf "\033[33m[..]\033[0m %s\n" "$1"; }
red() { printf "\033[31m[!!]\033[0m %s\n" "$1" >&2; }

mkdir -p "$OUT_DIR"

# Pre-flight summary so the operator can sanity-check what's about to ship.
yellow "snapshot pre-flight"
echo "  repo:        $PROJECT_ROOT"
echo "  out:         $ARCHIVE"
echo "  phase tag:   $PHASE_TAG"
echo "  git HEAD:    $(git rev-parse --short HEAD 2>/dev/null || echo '(no git)')"
DIRTY_FILES="$(git status --short 2>/dev/null | wc -l || echo 0)"
echo "  uncommitted: ${DIRTY_FILES} files (will be packed verbatim; review with 'git status' on destination)"
echo

# Build the tarball. Run from PROJECT_ROOT, archive paths relative to repo.
yellow "creating tar.gz (compressing — should take 1-3 s for typical phase sizes; longer if .npz files are heavy)"

# Build the exclude list. Heavy excludes only when the operator opts in via --no-heavy.
EXCLUDES=(
  --exclude='./models'
  --exclude='./data'
  --exclude='./.venv'
  --exclude='./.env'
  --exclude='./__pycache__'
  --exclude='*/__pycache__'
  --exclude='*.pyc'
  --exclude='./runs_setup.log'
  --exclude='./runs_setup_a800.log'
  --exclude='./recollection-patch.tar.gz'
  --exclude="${OUT_DIR#$PROJECT_ROOT/}"
  # LaTeX build artifacts in case anyone compiled the draft locally.
  --exclude='./_draft.aux'
  --exclude='./_draft.bbl'
  --exclude='./_draft.blg'
  --exclude='./_draft.log'
  --exclude='./_draft.out'
  --exclude='./_draft.toc'
  --exclude='./_draft.synctex.gz'
)
if [[ "$NO_HEAVY" == "--no-heavy" ]]; then
  yellow "  --no-heavy: excluding runs/**/*.npz from this snapshot"
  EXCLUDES+=( --exclude='./runs/*.npz' --exclude='./runs/**/*.npz' )
fi

tar "${EXCLUDES[@]}" -czf "$ARCHIVE" -C "$PROJECT_ROOT" .

SIZE_BYTES="$(stat -c%s "$ARCHIVE" 2>/dev/null || stat -f%z "$ARCHIVE")"
SIZE_HUMAN="$(numfmt --to=iec --suffix=B "$SIZE_BYTES" 2>/dev/null || echo "${SIZE_BYTES} bytes")"
SHA256="$(sha256sum "$ARCHIVE" 2>/dev/null | awk '{print $1}' || shasum -a 256 "$ARCHIVE" | awk '{print $1}')"

green "snapshot built"
echo "  path:        $ARCHIVE"
echo "  size:        $SIZE_HUMAN"
echo "  sha256:      $SHA256"
echo

yellow "verifying contents (top-level dirs + summary)"
# Use awk single-pass to avoid SIGPIPE-from-head on large archives.
tar -tzf "$ARCHIVE" | awk '
  { entries++ }
  /^\.\/[^/]+\/?$/ { top[$0]++ }
  /^\.\/runs\/[^/]+$/ { runs++ }
  /^\.\/figs\/[^/]+$/ { figs++ }
  /^\.\/scripts\/[^/]+$/ { scripts++ }
  /^\.\/tools\/[^/]+$/ { tools++ }
  /^\.\/extras\/[^/]+$/ { extras++ }
  /^\.\/\.git\// { git++ }
  END {
    print "  total entries:       " entries
    print "  .git/ entries:       " git
    print "  runs/ files:         " (runs+0)
    print "  figs/ files:         " (figs+0)
    print "  scripts/ files:      " (scripts+0)
    print "  tools/ files:        " (tools+0)
    print "  extras/ files:       " (extras+0)
    print "  top-level entries:"
    for (k in top) print "      " k
  }
' | sort
echo

# Restoration instructions printed inline so the operator doesn't need to
# go digging when they land on the destination machine.
cat <<EOF
$(green 'restore on destination machine:')

  mkdir -p /path/to/local/${REPO_NAME}
  cd /path/to/local/${REPO_NAME}
  tar -xzf ${ARCHIVE##*/}                   # extracts repo at current dir
  git status                                 # see uncommitted phase-N work
  git log --oneline -5                       # confirm history matches sha
  bash scripts/setup.sh                      # rebuild conda env (~3 min)
  bash scripts/download_models.sh phase1.4   # re-pull models (SHAs pinned)

  Verify integrity:
    echo "${SHA256}  ${ARCHIVE##*/}" | sha256sum -c
EOF
