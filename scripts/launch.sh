#!/bin/bash

encoder=SAM
bsize=5000
# bsize=10000

saveto=$1

lns=$(wc -l artifacts/20253_wbu.txt | awk ' { print $1 }')
echo $lns

for (( i=0; i<=lns; i+=bsize ))
do
    echo $i
    # qsub -cwd -N emb -l A6000,gpu,highp,cuda=1,h_rt=16:00:00 -o logs -j y \
    qsub -cwd -N emb -l RTX2080Ti,gpu,cuda=1,h_rt=23:00:00 -o logs -j y \
        ./scripts/job.sh $encoder $i $bsize $saveto $2 $3 $4
done
