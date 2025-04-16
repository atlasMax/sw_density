#!/bin/bash

# Activate virtual environment
source ../.venv/bin/activate

SESSION_NAME="compile_tints"

# Set a path for your log file
LOG_FILE="logs/sw_tint_compile$(date +%Y%m%d_%H%M%S).txt"

# Command to run
CMD="python -u compile_tints.py > $LOG_FILE 2>&1"


# Check if the screen session already exists
if screen -list | grep -q "$SESSION_NAME"; then
    echo "Screen session '$SESSION_NAME' is already running."
else
    # Start a detached screen session that runs the command
    screen -dmS $SESSION_NAME bash -c "$CMD"
    echo "Logging output to '$LOG_FILE'."
    echo "Use 'screen -r $SESSION_NAME' to reattach."
fi