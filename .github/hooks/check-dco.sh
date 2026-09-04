#!/usr/bin/env bash
# DCO sign-off check for the pre-commit commit-msg stage.
#
# Mirrors the GitHub DCO app requirement: every commit must carry a
# "Signed-off-by:" line identifying the author.
#
# Usage: check-dco.sh <commit-message-file>

set -euo pipefail

msg_file="${1:-}"

if [[ -z "${msg_file}" || ! -f "${msg_file}" ]]; then
    echo "DCO check: no commit message file supplied." >&2
    exit 1
fi

if grep -qE '^Signed-off-by: .+ <[^@ ]+@[^@ ]+>$' "${msg_file}"; then
    exit 0
fi

cat >&2 <<'EOF'
DCO check failed: commit message is missing a "Signed-off-by:" line.

Add a sign-off using one of:
    git commit -s            # sign as you create the commit
    git commit --amend -s    # sign the most recent commit

The line must identify the commit author, for example:
    Signed-off-by: Your Name <you@example.com>
EOF

exit 1
