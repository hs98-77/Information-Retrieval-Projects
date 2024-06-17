#!/bin/bash
let i=1
for lambda in 0.3 0.35 0.4 0.45 0.5 0.55 0.6; 
do
  ~/galago/galago-3.16/core/target/appassembler/bin/galago batch-search --lambda=$lambda JM.json > $i.txt
  i="$(($i+1))"
done