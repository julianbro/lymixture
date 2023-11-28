#!/bin/bash

source .venv/bin/activate
# Find the path to the kernel.json file
kernel_file=$(python3.10 -c "from jupyter_client import find_connection_file; print(find_connection_file())")

# Check if kernel_file is empty (i.e., kernel.json not found)
if [[ -z $kernel_file ]]; then
    echo "kernel.json not found. Make sure you have a Jupyter kernel running."
    exit 1
fi

# Run jupyter console with the --existing flag
jupyter console --existing "$kernel_file"