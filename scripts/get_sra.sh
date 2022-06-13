#!/bin/bash
input=$1
output=$2
while read -r line
do
	echo $line
	fasterq-dump --outfile "$output"/"$line" --split-3 --threads 8 --skip-technical $line
done < $input
