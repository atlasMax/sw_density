#!/bin/bash

# Activate virtual environment
source ../.venv/bin/activate


SESSION_NAME="corr_facs_full"

# Set a path for your log file
LOG_FILE="logs/correction_factors_FULL_log_$(date +%Y%m%d_%H%M%S).txt"

NE_CAL_PATH="output_data_new_2/"
NUM_WORKERS=30
# Command to run
CMD="python -u run_correction_factors_parallel_FULL_BVR.py $NE_CAL_PATH $NUM_WORKERS > $LOG_FILE 2>&1"


# Check if the screen session already exists
if screen -list | grep -q "$SESSION_NAME"; then
    echo "Screen session '$SESSION_NAME' is already running."
else
    # Start a detached screen session that runs the command
    screen -dmS $SESSION_NAME bash -c "$CMD"
    echo "Calculating correction factors with $NUM_WORKERS workers."
    echo "Reading .cdf files from '$NE_CAL_PATH' and writing to '$NE_CAL_PATH stats/'"
    echo "Logging output to '$LOG_FILE'."
fi