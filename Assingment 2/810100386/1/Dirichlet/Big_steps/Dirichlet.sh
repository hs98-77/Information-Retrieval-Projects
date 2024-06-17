#!/bin/bash
let i=1
for ((mu=0;mu<=3500;mu=$mu+50)); 
do
  ~/galago/galago-3.16/core/target/appassembler/bin/galago batch-search --mu=$mu Dirichlet.json > $i.txt
  i="$(($i+1))"
done
