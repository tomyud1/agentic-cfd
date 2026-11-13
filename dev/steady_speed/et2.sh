A=~/OpenFOAM/tomyud-v2606/run/agentic
at() { grep ExecutionTime $1 | sed -n "${2}p" | awk '{print $3}'; }
echo "N128 a0.9 (case 2) to row 1931: $(at $A/lid_driven_cavity_re100/N128/log.simpleFoam 1931) s"
for r in s95:931 s97:571 s98:411 s99:751 seq_s97:471; do n=${r%%:*}; k=${r##*:}; echo "N128 $n to row $k: $(at $A/dev_steady/$n/log.simpleFoam $k) s"; done
echo "N256 a0.9 (case 2) to row 7491: $(at $A/lid_driven_cavity_re100/N256/log.simpleFoam 7491) s; full run $(at $A/lid_driven_cavity_re100/N256/log.simpleFoam 10864) s"
for r in s97_256:2121 seq_s97_256:981 s99_256:801 seq_s99_256:1241 s995_256:1411 live_s99_256:815; do n=${r%%:*}; k=${r##*:}; echo "N256 $n to row $k: $(at $A/dev_steady/$n/log.simpleFoam $k) s"; done
echo "N512 seq_s99_512 to row 1441: $(at $A/dev_steady/seq_s99_512/log.simpleFoam 1441) s (2 runs sharing CPU)"
