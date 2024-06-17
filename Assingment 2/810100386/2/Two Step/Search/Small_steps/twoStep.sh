#!/bin/bash
let i=1
let j=1
for ((mu=1000;mu<=1400;mu=$mu+40)); 
do
  j=0
  for lambda1 in 0 0.03 0.06 0.09 0.12 0.15 0.18 0.21 0.24 0.27 0.3;
  do
    ~/galago/galago-3.16/core/target/appassembler/bin/galago batch-search --lambda1=$lambda1 --mu=$mu twoStep.json > $i.$j.txt
    j="$(($j+1))"
  done
  i="$(($i+1))"
done
