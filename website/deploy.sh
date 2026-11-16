#!/usr/bin/env bash
# Build the course site and deploy it to Vercel as plain static files.
#
# No Git integration: the Vercel project is created through the API without a
# repository, and the upload happens from a temporary folder outside this Git
# checkout, so nothing is linked to any GitHub account. The Vercel CLI runs
# with its own throwaway config folder, so an existing `vercel login` on this
# machine is never used or changed.
#
# Configuration (environment variables, or website/.env, which is gitignored):
#   VERCEL_TOKEN        required  token from the classroom Vercel account
#   VERCEL_PROJECT      optional  project name (default: nom-ocr-ws2026)
#   VERCEL_TEAM_ID      optional  team ID if the project lives in a team
#   DOCS_SOURCE_REF     optional  Git ref that notebook/script links point to (default: main)
#
# Usage:  website/deploy.sh            # production deployment
#         website/deploy.sh --preview  # preview URL only
#         website/deploy.sh --build-only
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
[[ -f .env ]] && { set -a; source .env; set +a; }

MODE="${1:---prod}"
PROJECT="${VERCEL_PROJECT:-nom-ocr-ws2026}"
API="https://api.vercel.com"
TEAM_QS="${VERCEL_TEAM_ID:+?teamId=$VERCEL_TEAM_ID}"

echo "==> Building site (links point to ref: ${DOCS_SOURCE_REF:-main})"
uv run --project "$HERE" mkdocs build --strict --clean --config-file "$HERE/mkdocs.yml" --site-dir "$HERE/site"
[[ "$MODE" == "--build-only" ]] && { echo "Built: $HERE/site"; exit 0; }

: "${VERCEL_TOKEN:?Set VERCEL_TOKEN (classroom Vercel account) in the environment or website/.env}"
auth=(-H "Authorization: Bearer $VERCEL_TOKEN")

echo "==> Looking up Vercel project '$PROJECT'"
lookup="$(curl -sS "${auth[@]}" "$API/v9/projects/$PROJECT$TEAM_QS")"
project_id="$(printf '%s' "$lookup" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("id",""))')"
if [[ -z "$project_id" ]]; then
  echo "==> Creating project '$PROJECT' (static, no Git repository)"
  created="$(curl -sS "${auth[@]}" -H "Content-Type: application/json" -X POST "$API/v11/projects$TEAM_QS" \
    -d "{\"name\": \"$PROJECT\", \"framework\": null}")"
  project_id="$(printf '%s' "$created" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("id") or "")')"
  [[ -n "$project_id" ]] || { echo "Project creation failed: $created" >&2; exit 1; }
  lookup="$created"
fi
git_link="$(printf '%s' "$lookup" | python3 -c 'import json,sys; print((json.load(sys.stdin).get("link") or {}).get("type",""))')"
if [[ -n "$git_link" ]]; then
  echo "Refusing to deploy: project '$PROJECT' is connected to a Git provider ($git_link)." >&2
  exit 1
fi
org_id="${VERCEL_TEAM_ID:-$(printf '%s' "$lookup" | python3 -c 'import json,sys; print(json.load(sys.stdin)["accountId"])')}"

stage="$(mktemp -d)"; cli_home="$(mktemp -d)"
trap 'rm -rf "$stage" "$cli_home"' EXIT
cp -R "$HERE/site/." "$stage/"
mkdir -p "$stage/.vercel"
printf '{"orgId": "%s", "projectId": "%s"}\n' "$org_id" "$project_id" > "$stage/.vercel/project.json"
printf '{"git": {"deploymentEnabled": false}}\n' > "$stage/vercel.json"

flags=(--yes --token "$VERCEL_TOKEN" --global-config "$cli_home")
[[ "$MODE" == "--prod" ]] && flags+=(--prod)
echo "==> Uploading static site from $stage"
cd "$stage"
VERCEL_TELEMETRY_DISABLED=1 npx --yes vercel@latest deploy "${flags[@]}"
