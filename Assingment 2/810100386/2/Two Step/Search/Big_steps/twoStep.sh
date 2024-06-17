#!/bin/bash
let i=1
let j=1
for ((mu=0;mu<=2000;mu=$mu+200)); 
do
  j=0
  for lambda1 in 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1;
  do
    ~/galago/galago-3.16/core/target/appassembler/bin/galago batch-search --lambda1=$lambda1 --mu=$mu twoStep.json > $i.$j.txt
    j="$(($j+1))"
  done
  i="$(($i+1))"
done
