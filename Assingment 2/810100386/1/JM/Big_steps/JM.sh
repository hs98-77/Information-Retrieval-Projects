#!/bin/bash
let i=1
for lambda in 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1; 
do
  ~/galago/galago-3.16/core/target/appassembler/bin/galago batch-search --lambda=$lambda JM.json > $i.txt
  i="$(($i+1))"
done