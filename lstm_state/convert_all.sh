#!/bin/bash


if [ $# -lt 1 ]; then
    echo "Usage: ./convert_all.sh <stage>"
    exit 1
fi

stage=$1

num_chunks=48
let max_chunk=num_chunks-1

seq 0 $max_chunk | xargs -P $num_chunks -I % python \
    lstm_state/convert_to_tfrecord.py --stage $stage --chunk %

