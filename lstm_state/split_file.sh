#!/usr/bin/env bash

# split a training / validation files into chunks

tempfile=mktemp

if [ $# -lt 2 ]; then
    echo "Usage: ./split_file.sh <filename> <num_chunks>"
    exit 1
fi

inputfile=$1
num_chunks=$2

echo "Splitting file $inputfile"

cat $inputfile | tr ' ' '\n'  > $tempfile
split -d -n $num_chunks $tempfile $inputfile

rm $tempfile

