#!/bin/bash
# Script to run all tests from the build directory
# Usage: ./run_tests.sh [test_name]
# If no test_name is provided, runs all tests

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/build"

# Use the correct Python interpreter from the chesshacks conda environment
PYTHON="python3"

if [ -z "$1" ]; then
    echo "Running all tests..."
    echo "===================="
    
    for test in tests/*.py; do
        echo ""
        echo "Running $test..."
        echo "----------------------------------------"
        $PYTHON "$test"
        if [ $? -ne 0 ]; then
            echo "❌ Test failed: $test"
            exit 1
        fi
    done
    
    echo ""
    echo "===================="
    echo "✓ All tests passed!"
else
    echo "Running test: $1"
    $PYTHON "tests/$1"
fi
