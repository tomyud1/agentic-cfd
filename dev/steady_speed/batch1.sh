D=$HOME/OpenFOAM/tomyud-v2606/run/agentic/dev_steady; mkdir -p $D; rm -f $D/times.txt
X=/mnt/c/Users/tomyu/AppData/Local/Temp/claude/89cc76715156c6c4/029ac6cd-ed6b-4f07-a4fe-677bcd80aa77/scratchpad/exp.sh
bash $X s95 simple 0.95 & bash $X s97 simple 0.97 & bash $X pc5 pimple 5 & bash $X pc20 pimple 20 & wait
bash $X pc50 pimple 50 & bash $X seq_s90 simple 0.9 N64 & bash $X seq_pc20 pimple 20 N64 & wait
echo BATCH1_DONE >> $D/times.txt
