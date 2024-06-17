#!/bin/bash
let i=1
for ((mu=1550;mu<=1900;mu=$mu+10)); 
do
  ~/galago/galago-3.16/core/target/appassembler/bin/galago batch-search --mu=$mu Dirichlet.json > $i.txt
  i="$(($i+1))"
done
