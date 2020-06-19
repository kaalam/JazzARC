#!/bin/bash

rm -rf src/__pycache__/
rm -rf src/*.pyc
rm -rf .ipynb_checkpoints/
rm -rf .pytest_cache
rm -rf .cache/
rm -rf .coverage
rm -rf htmlcov/
rm -rf kagglespace/
rm -f *.ipynb
rm -f test_*.jcb
find . | grep -e '/\.pytest' | xargs rm -rf
