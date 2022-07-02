#!/bin/bash

SHORT=d:,i:,h
LONG=delete:,in_dir:,help
OPTS=$(getopt --options $SHORT --longoptions $LONG -- "$@")

eval set -- "$OPTS"

while :
do
  case "$1" in
    -d | --delete )
      delete="$2"
      shift 2
      ;;
    -i | --in_dir )
      in_dir="$2"
      shift 2
      ;;
    -h | --help)
      "This is a profiling script"
      exit 2
      ;;
    --)
      shift;
      break
      ;;
    *)
      echo "Unexpected option: $1"
      ;;
  esac
done

for sra in $in_dir/*
do
	file_name="${sra%%.*}"
	bn="${file_name##*/}"
	echo $bn
	if  [ "$(ls $sra | wc -l)" -eq 2 ]
	then
		motus profile -f $sra/"$bn"_1.fastq -r $sra/"$bn"_2.fastq -t 25 -g 1 -C precision > $sra/"$bn".profile 
	elif [ "$(ls $sra | wc -l)" -eq 1 ]
	then
		motus profile -s $sra/"$bn".fastq -t 25 -g 1 -C precision > $sra/"$bn".profile 
	else
		echo "something is wrong"
		echo $sra
	fi
done

