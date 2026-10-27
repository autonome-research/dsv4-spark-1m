#!/usr/bin/env bash
# Copy the canonical reports from ~/reports into reports/, then commit everything
# that changed in the repository.
# Usage: tools/commit-snapshot.sh "message"
set -euo pipefail
cd /srv/dsv4/run
cp -p "$HOME/reports/DSV4 optimisation log.md" "$HOME/reports/DSV4 final report.md" reports/
git add -A
git diff --cached --quiet && { echo "nothing changed"; exit 0; }
git commit -q -m "${1:?commit message}" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git log --oneline -1
