#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# container: docker.io/cphsieh/ruler:0.1.0
# bash run.sh MODEL_NAME BENCHMARK_NAME

if [ $# -ne 8 ]; then
    echo "Usage: $0 <model_type> <fms_name> <disp_name> <model_directory> <tokenizer_type> <benchmark_name> <sequence_length> <batch_size>"
    echo "  model_type: fms or mamba_ssm"
    exit 1
fi


# Root Directories
ROOT_DIR="/gpfs/hshen/RULER" # the path that stores generated task samples and model predictions.
ENGINE_DIR="." # the path that contains individual engine folders from TensorRT-LLM.
MODEL_TYPE=${1} # fms or mamba
if [ "${MODEL_TYPE}" == "fms" ]; then
    MODEL_NAME="custom-fms"
elif [ "${MODEL_TYPE}" == "mamba_ssm" ]; then
    MODEL_NAME="custom-mamba-fms"
else
    echo "Error: model_type must be 'fms' or 'mamba_ssm', got '${MODEL_TYPE}'"
    exit 1
fi
FMS_NAME=${2} # FMS configuration name
DISPLAY_NAME=${3}
MODEL_DIR=${4} # the path that contains individual model folders from Huggingface.
TOKENIZER=${5} # Make sure generation is not done repeatedly for the same tokenizer
BENCHMARK=${6}
SEQ_LENGTHS=${7}
BATCH_SIZE=${8}
GPUS=${GPUS:-8} # default to all 8 GPUs; override with GPUS=N

# Model and Tokenizer
source config_models.sh
MODEL_CONFIG=$(MODEL_SELECT ${MODEL_NAME} ${MODEL_DIR} ${ENGINE_DIR})
IFS=":" read MODEL_PATH MODEL_TEMPLATE_TYPE MODEL_FRAMEWORK _TOKENIZER_PATH TOKENIZER_TYPE OPENAI_API_KEY GEMINI_API_KEY AZURE_ID AZURE_SECRET AZURE_ENDPOINT <<< "$MODEL_CONFIG"
if [ -z "${MODEL_PATH}" ]; then
    echo "Model: ${MODEL_NAME} is not supported"
    exit 1
fi

# Use explicit tokenizer path instead of auto-detected one
TOKENIZER_PATH="/gpfs/hshen/tokenizer/${TOKENIZER}"


export OPENAI_API_KEY=${OPENAI_API_KEY}
export GEMINI_API_KEY=${GEMINI_API_KEY}
export AZURE_API_ID=${AZURE_ID}
export AZURE_API_SECRET=${AZURE_SECRET}
export AZURE_API_ENDPOINT=${AZURE_ENDPOINT}


# Benchmark and Tasks
source config_tasks.sh
declare -n TASKS=$BENCHMARK
if [ -z "${TASKS}" ]; then
    echo "Benchmark: ${BENCHMARK} is not supported"
    exit 1
fi

# Resolve the actual data benchmark (e.g. niah_single -> synthetic)
# Subsets define <name>_benchmark in config_tasks.sh; fall back to BENCHMARK itself
BENCHMARK_VAR="${BENCHMARK}_benchmark"
DATA_BENCHMARK=${!BENCHMARK_VAR:-$BENCHMARK}

# Start client (prepare data / call model API / obtain final metrics)
total_time=0
for MAX_SEQ_LENGTH in "${SEQ_LENGTHS[@]}"; do

    # Modified the data generation logic here: make the generation consistent for all models
    DATA_DIR="${ROOT_DIR}/data/${TOKENIZER}/${DATA_BENCHMARK}/${MAX_SEQ_LENGTH}"
    # DATA_DIR="${ROOT_DIR}/data_dliu/${TOKENIZER}/${DATA_BENCHMARK}/${MAX_SEQ_LENGTH}"
    PRED_DIR="${ROOT_DIR}/fms/${DISPLAY_NAME}/${BENCHMARK}/${MAX_SEQ_LENGTH}/pred"
    mkdir -p ${DATA_DIR}
    mkdir -p ${PRED_DIR}

    for TASK in "${TASKS[@]}"; do
        python data/prepare.py \
            --save_dir ${DATA_DIR} \
            --benchmark ${DATA_BENCHMARK} \
            --task ${TASK} \
            --tokenizer_path ${TOKENIZER_PATH} \
            --tokenizer_type ${TOKENIZER_TYPE} \
            --max_seq_length ${MAX_SEQ_LENGTH} \
            --model_template_type ${MODEL_TEMPLATE_TYPE} \
            --num_samples ${NUM_SAMPLES} \
            ${REMOVE_NEWLINE_TAB}

        start_time=$(date +%s)
        accelerate launch --num_processes ${GPUS} pred/call_api.py \
            --data_dir ${DATA_DIR} \
            --save_dir ${PRED_DIR} \
            --benchmark ${DATA_BENCHMARK} \
            --task ${TASK} \
            --server_type ${MODEL_FRAMEWORK} \
            --model_name_or_path ${MODEL_PATH} \
            --tokenizer_path ${TOKENIZER_PATH} \
            --temperature ${TEMPERATURE} \
            --top_k ${TOP_K} \
            --top_p ${TOP_P} \
            --batch_size ${BATCH_SIZE} \
            --fms_variant ${FMS_NAME} \
            --use_accelerate \
            ${STOP_WORDS}
        end_time=$(date +%s)
        time_diff=$((end_time - start_time))
        total_time=$((total_time + time_diff))
    done

    python eval/evaluate.py \
        --data_dir ${PRED_DIR} \
        --benchmark ${DATA_BENCHMARK}
done

echo "Total time spent on call_api: $total_time seconds"