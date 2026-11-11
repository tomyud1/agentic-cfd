"""One-off migration of cases 1-2 to the contract of 2026-11-10 (decision kinds, insight ids/times)."""
from pathlib import Path

from cfdlib import insights

ROOT = Path(__file__).resolve().parents[1]
KINDS = {
    "plane_poiseuille": ["physics", "solver", "physics", "physics", "solver", "physics", "accuracy", "accuracy",
                         "solver", "solver", "accuracy", "output"],
    "lid_driven_cavity_re100": ["physics", "solver", "physics", "physics", "solver", "accuracy", "accuracy",
                                "solver", "solver", "accuracy", "output"],
}

for case, kinds in KINDS.items():
    p = ROOT / "cases" / case / "decisions.yaml"
    lines = p.read_text(encoding="utf-8").splitlines()
    if not any(l.startswith("  kind:") for l in lines):
        out, i = [], 0
        for l in lines:
            out.append(l)
            if l.startswith("- area:"):
                out.append(f"  kind: {kinds[i]}")
                i += 1
        assert i == len(kinds), (case, i)
        p.write_text("\n".join(out) + "\n", encoding="utf-8")

    entries = insights.load(case)
    for n, e in enumerate(entries, 1):
        new = dict(id=n, stage=e["stage"], logged=e.get("logged", "2026-11-10 (time not recorded)"), status="resolved")
        new.update({k: v for k, v in e.items() if k not in new})
        entries[n - 1] = new
    insights._save(case, entries)
    print(case, len(kinds), "decisions,", len(entries), "insights")
