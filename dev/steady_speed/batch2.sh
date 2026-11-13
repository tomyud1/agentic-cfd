cd /tmp
D=$HOME/OpenFOAM/tomyud-v2606/run/agentic/dev_steady
X=/mnt/c/Users/tomyu/AppData/Local/Temp/claude/89cc76715156c6c4/029ac6cd-ed6b-4f07-a4fe-677bcd80aa77/scratchpad/exp.sh
bash $X s98 simple 0.98 & bash $X s99 simple 0.99 & bash $X s995 simple 0.995 & bash $X seq_s97 simple 0.97 N64 & wait
bash $X s97_256 simple 0.97 rest 256 4000 & bash $X seq_s97_256 simple 0.97 N128 256 4000 & bash $X s99_256 simple 0.99 rest 256 4000 & wait
echo BATCH2_DONE >> $D/times.txt
