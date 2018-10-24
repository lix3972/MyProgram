python3 train_Donly.py --model=D0 --warpN=0 --pertFG=0.2
a="python3 train_STGAN.py --model=test0 --loadD=0/D0_warp0_it50000 --warpN="
for ((w=1;w<=5;w++)) do cmd=${a}${w}; eval $cmd; done;
