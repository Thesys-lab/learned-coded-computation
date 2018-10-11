#!/bin/bash

NUMARG=1
if [ "$#" -ne "$NUMARG" ]; then
  echo "Usage: $0 <to_format>"
  echo "  to_format: file or directory to format. If a directory is entered,"
  echo "             all .py files in that directory (and subdirectories) will"
  echo "             be formatted."
  exit 1
fi

find $1 -name '*.py' -exec autopep8 --in-place '{}' \;
