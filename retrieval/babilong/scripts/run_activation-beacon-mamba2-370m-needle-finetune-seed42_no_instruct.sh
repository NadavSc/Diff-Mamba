#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=0 ./script_name.sh
set -e

RESULTS_FOLDER="./babilong_evals"
MODEL_NAME="mamba2-370m-needle-finetune-seed42"
MODEL_PATH="../../../outputs/mamba2-370m-needle-finetune-seed42"
TOKENIZER_PATH="../../../outputs/mamba2-370m-needle-finetune-seed42"


MAMBA=true
LOCAL=true
DIFF=false
USE_CHAT_TEMPLATE=false
USE_INSTRUCTION=false
USE_EXAMPLES=false
USE_POST_PROMPT=false
API_URL=""

DATASET_NAME="RMT-team/babilong-1k-samples"
TASKS=("qa1" "qa2" "qa3" "qa4" "qa5")
LENGTHS=("1k" "2k" "4k" "8k" "16k" "32k" "64k")

echo running $MODEL_PATH on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_model_on_babilong.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_path "$MODEL_PATH" \
    --tokenizer_path "$TOKENIZER_PATH" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$LOCAL" == true ] && echo "--local" ) \
    $( [ "$MAMBA" == true ] && echo "--mamba" ) \
    $( [ "$DIFF" == true ] && echo "--diff" ) \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    --api_url "$API_URL"


EVAL0K=true
TASKS=("qa1" "qa2" "qa3" "qa4" "qa5")
LENGTHS=("0k")

echo running $MODEL_PATH on "${TASKS[@]}" with "${LENGTHS[@]}"

python run_model_on_babilong.py \
    --results_folder "$RESULTS_FOLDER" \
    --dataset_name "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --model_path "$MODEL_PATH" \
    --tokenizer_path "$TOKENIZER_PATH" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    $( [ "$LOCAL" == true ] && echo "--local" ) \
    $( [ "$MAMBA" == true ] && echo "--mamba" ) \
    $( [ "$DIFF" == true ] && echo "--diff" ) \
    $( [ "$EVAL0K" == true ] && echo "--diff" ) \
    $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
    $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
    $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
    $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
    --api_url "$API_URL"
