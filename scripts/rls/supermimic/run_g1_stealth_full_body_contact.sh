#!/usr/bin/env bash
set -euo pipefail
test "$(git branch --show-current)" = main
export SUPERMIMIC_STEALTH_RECIPE=fullbody-contact
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_g1_stealth_combined.sh" "$@"
