#!/bin/bash


SCRIPT="finetune.py"

for MODEL in  "llama3.1-8b" # "llama2-13b"
do
    for DATASET in  "3way" "2way"
    do
        OUTPUT_PATH="experiments/ir-tuning-short-answer-${DATASET}/ir-tuning-short-answer-${MODEL}-lora-full"
        LOG_FILE="${OUTPUT_PATH}/log.txt"

        mkdir -p "$OUTPUT_PATH"
        echo "Running: model=$MODEL dataset=$DATASET"

        python $SCRIPT \
            --model-name "$MODEL" \
            --dataset-name "$DATASET" \
            --output-dir "$OUTPUT_PATH" \
            2>&1 | tee "$LOG_FILE"
    done
done