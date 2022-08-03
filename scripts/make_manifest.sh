#!/bin/bash
input_dir=$1
manifest="$input_dir/manifest.txt"
echo "sample-id,absolute-filepath,direction" > $manifest
for dir in $input_dir/* 
do
	#dir_name="${dir%%.*}"
        id="${dir##*/}"
	#echo $id
	printf '%s,%s,%s\n' "$id" "$PWD"/"$dir"/"$id"_1.fastq "forward" >> $manifest
	printf '%s,%s,%s\n' "$id" "$PWD"/"$dir"/"$id"_2.fastq "reverse" >> $manifest
done
sed -i '/manifest/d' $manifest
