#!/usr/bin/env bash
# 部署后冒烟测试：./smoke.sh https://demo.example.com   （本机试跑：./smoke.sh https://localhost -k）
set -u
BASE="${1:?usage: smoke.sh BASE_URL [curl flags]}"; shift || true
C=(curl -sS -m 150 "$@")
fail=0
ok()   { echo "  ok    $1"; }
bad()  { echo "  FAIL  $1"; fail=1; }
check(){ if eval "$2"; then ok "$1"; else bad "$1"; fi; }

echo "health / page"
H=$("${C[@]}" "$BASE/api/health")
check "health ok and demo mode on"         '[[ "$H" == *"\"status\":\"ok\""* && "$H" == *"\"demo\":true"* ]]'
P=$("${C[@]}" "$BASE/")
check "page served"                         '[[ "$P" == *"Data Discovery System"* ]]'
check "page has no telemetry scripts"       '[[ "$P" != *"uba-script"* && "$P" != *"apm-insight"* && "$P" != *"cowork.xiaohongshu.com"* ]]'
check "page flagged as demo"                '[[ "$P" == *"DD_DEMO=true"* ]]'

echo "blocked paths"
for path in /api/cookie/fetch /api/me /api/pairs /api/prompts/selection /docs /openapi.json; do
  code=$("${C[@]}" -o /dev/null -w '%{http_code}' "$BASE$path"); check "$path -> 404" '[[ "$code" == "404" ]]'
done
code=$("${C[@]}" -o /dev/null -w '%{http_code}' -X POST "$BASE/api/cookie/diagnose" -H 'content-type: application/json' -d '{}'); check "POST /api/cookie/diagnose -> 404" '[[ "$code" == "404" ]]'

echo "example questions (stage 5 must select a primary asset)"
for q in "Last 7 days DAU and CTR by platform" "Last 30 days CTR by platform" "Last 7 days DAU and GMV by channel"; do
  R=$("${C[@]}" -X POST "$BASE/api/stage5" -H 'content-type: application/json' -d "{\"query\":\"$q\"}")
  check "$q" 'echo "$R" | python3 -c "import sys,json; d=json.load(sys.stdin); s=d[\"selection\"]; sys.exit(0 if s[\"primary\"] and not any(f.startswith((\"no_asset_selected\",\"metrics_not_covered\")) for f in s[\"risk_flags\"]) else 1)"'
done

echo "limits"
code=$("${C[@]}" -o /dev/null -w '%{http_code}' -X POST "$BASE/api/stage1" -H 'content-type: application/json' -d "{\"query\":\"$(head -c 3000 /dev/zero | tr '\0' x)\"}"); check "overlong query rejected (4xx)" '[[ "$code" == 4* ]]'
[[ $fail == 0 ]] && echo "ALL OK" || { echo "SOME CHECKS FAILED"; exit 1; }
