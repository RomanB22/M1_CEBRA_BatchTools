#!/bin/bash

#echo "Actual Path: $PWD"
unset DISPLAY
export PYTHONPATH="${PYTHONPATH}:$PWD"
echo "PYTHONPATH updated"
echo $PYTHONPATH