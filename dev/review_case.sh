#!/bin/bash
# First pass of the dev agent's review of a solved case (dev/LOOP.md, step 4): the checks a script
# can do. Run on the Linux side, from anywhere:
#   bash dev/review_case.sh <case folder name> [finest case dir]
# It changes nothing. What it cannot check (is the physics right? is the verdict fair?) is the
# review checklist in dev/LOOP.md.
cd "$(dirname "$0")/.." || exit 1
PY="${PY:-$HOME/venvs/cfd/bin/python}"
c="$1"; fin="$2"
[ -f "cases/$c/problem.yaml" ] || { echo "usage: bash dev/review_case.sh <case> [finest case dir]   (no cases/$c/problem.yaml)"; exit 2; }

echo "== 1. report contract"
$PY -m cfdlib.contract "results/$c/results.json" $fin | head -8

echo "== 2. solver processes still running (should be none)"
$PY -m cfdlib.foam ps

echo "== 3. the given problem and its reference data, changed since the last commit (should be none)"
git status --short -- "cases/$c/problem.yaml" "cases/$c/reference"

echo "== 4. shared files the solver agent changed (should be none: tool changes are the dev agent's)"
git status --short -- cfdlib workflow gpucfd setup tests dev AGENTS.md CLAUDE.md | grep -v "__pycache__"

echo "== 5. verdict, answers and counts"
$PY - "$c" <<'EOF'
import collections, json, sys
r = json.load(open(f"results/{sys.argv[1]}/results.json"))
print("verdict:", r["verdict"], "| type:", r.get("comparison_type"))
for a in r["summary"]["answers"]:
    print(" ", a.get("letter"), "|", str(a.get("label"))[:60], "|", str(a.get("diff", ""))[:80])
ins = r.get("insights", [])
print("decisions:", len(r.get("decisions", [])), "| figures:", len(r.get("figures", [])), "| insights:", len(ins))
print("insights by stage:", dict(collections.Counter(i.get("stage") for i in ins)))
print("unresolved insights:", [i.get("id") for i in ins if i.get("status") != "resolved"])
EOF

echo "== 6. the insight log"
$PY -m cfdlib.insights "$c" list
