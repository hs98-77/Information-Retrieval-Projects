#!/bin/bash
let i=1
for ((delta=0;delta<=6;delta=$delta+1)); 
do
  ~/galago/galago-3.16/core/target/appassembler/bin/galago batch-search --delta=$delta AS.json > $i.txt
  i="$(($i+1))"
done
