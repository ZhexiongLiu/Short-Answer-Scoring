#!/bin/bash

MODEL="gpt-4.1"
DATASET="2way"
OUTPUT_PATH="experiments"
BATCH_SIZE=60
MAX_CONCURRENT=120
PROMPT_TEMPLATE="baseline"

for MODEL in "gpt-5.2" "gpt-5.4" "gpt-4.1" #
do
    for DATASET in "2way" # "3way"
    do
        for PROMPT_TEMPLATE in  "baseline" "rationale"
        do

          OUTPUT_PATH="experiments/${MODEL}_${DATASET}_${PROMPT_TEMPLATE}"
          LOG_FILE="${OUTPUT_PATH}/log.txt"

          mkdir -p "$OUTPUT_PATH"
          echo "Running: model $MODEL dataset $DATASET template $PROMPT_TEMPLATE"

          python main.py \
              --model "$MODEL" \
              --dataset "$DATASET" \
              --output_path "$OUTPUT_PATH" \
              --batch_size "$BATCH_SIZE" \
              --max_concurrent "$MAX_CONCURRENT" \
              --prompt_template "$PROMPT_TEMPLATE" \
              2>&1 | tee "$LOG_FILE"
        done
    done
done