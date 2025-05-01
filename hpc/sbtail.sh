#!/bin/bash

JOB_INFO=$(sbatch run_main.slurm)
JOB_ID=$(echo $JOB_INFO | awk '{print $4}')

if [[ -z "$JOB_ID" ]]; then
  echo "Failed to submit job. sbatch said:"
  echo "$JOB_INFO"
fi

OUT_FILE="$HOME/logs/${JOB_ID}.log"
echo "Submitted batch job $JOB_ID; now tailing $OUT_FILE"
while [[ ! -f $OUT_FILE ]]; do
  sleep 1
done

tail -f $OUT_FILE
