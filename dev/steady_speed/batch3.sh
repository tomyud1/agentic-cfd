cd /tmp
D=$HOME/OpenFOAM/tomyud-v2606/run/agentic/dev_steady
X=/mnt/c/Users/tomyu/AppData/Local/Temp/claude/89cc76715156c6c4/029ac6cd-ed6b-4f07-a4fe-677bcd80aa77/scratchpad/exp.sh
bash $X seq_s99_256 simple 0.99 N128 256 2000 & bash $X s995_256 simple 0.995 rest 256 2000 & wait
bash $X s99_512 simple 0.99 rest 512 2500 & bash $X seq_s99_512 simple 0.99 N256 512 2500 & wait
echo BATCH3_DONE >> $D/times.txt
