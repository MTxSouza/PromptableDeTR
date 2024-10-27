#!/bin/bash
# Get all test files.
PYTHON_TEST_FILES=$(find ./tests -name "test_*.py")

# Test each file individually.
for test_file in $PYTHON_TEST_FILES;
do
    # Get name of the current test file.
    CURRENT_TEST_FILE=$(basename $test_file)
    echo "Running tests in $CURRENT_TEST_FILE..."

    # Run the test.
    python -m unittest $test_file
done