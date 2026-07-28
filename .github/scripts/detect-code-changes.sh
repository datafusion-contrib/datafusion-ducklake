#!/usr/bin/env bash
#
# Decides whether a CI run needs to build and test, writing `code=true|false`
# to $GITHUB_OUTPUT for the calling job's steps to gate on.
#
# This exists instead of a `paths-ignore` filter on the workflow triggers.
# A workflow skipped by path filtering never creates its check runs, and a
# required status check is satisfied only by a check run that exists and
# succeeded — so a docs-only pull request would sit at "Expected — waiting for
# status to be reported" and could never merge. Letting the job start and
# short-circuiting inside it keeps every required check name reporting.
#
# Fails open: anything unexpected (API error, unknown event, empty file list)
# yields `code=true`, so an uncertain run builds rather than silently passing.

set -euo pipefail

# Paths that cannot affect the build. Everything else counts as code.
NON_CODE_RE='(\.md$|^LICENSE$|^\.gitignore$|^\.github/CODEOWNERS$)'

emit() {
  echo "code=$1" >>"$GITHUB_OUTPUT"
  echo "code=$1"
}

files=""
case "${GITHUB_EVENT_NAME:-}" in
  pull_request)
    pr=$(jq -r '.pull_request.number' "$GITHUB_EVENT_PATH")
    files=$(gh api --paginate "repos/${GITHUB_REPOSITORY}/pulls/${pr}/files" \
      --jq '.[].filename' 2>/dev/null || true)
    ;;
  push)
    before=$(jq -r '.before' "$GITHUB_EVENT_PATH")
    # All-zero `before` means a newly created ref, which has no diff base.
    if [[ "$before" =~ ^0+$ ]]; then
      echo "New ref with no diff base — building."
      emit true
      exit 0
    fi
    files=$(gh api --paginate \
      "repos/${GITHUB_REPOSITORY}/compare/${before}...${GITHUB_SHA}" \
      --jq '.files[].filename' 2>/dev/null || true)
    ;;
  *)
    echo "Unhandled event '${GITHUB_EVENT_NAME:-}' — building."
    emit true
    exit 0
    ;;
esac

if [[ -z "$files" ]]; then
  echo "Could not determine changed files — building."
  emit true
  exit 0
fi

echo "Changed files:"
printf '%s\n' "$files" | sed 's/^/  /'

# grep -q -v exits 0 as soon as one path is *not* in the non-code set.
if printf '%s\n' "$files" | grep -qvE "$NON_CODE_RE"; then
  emit true
else
  echo "Docs and repository metadata only — skipping build and tests."
  emit false
fi
