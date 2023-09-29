#!/bin/bash

# 检查参数
if [ "$1" == "c" ] || [ -z "$1" ]; then
  echo "Clean all files in the./fig,./log,./model directory (except.gitkeep)..."
  
  for dir in ./fig ./log ./model; do
    find "$dir" -mindepth 1 ! -name '.gitkeep' -exec rm -rf {} +
  done
fi

if [ "$1" == "l" ] || [ -z "$1" ]; then
  # echo "run python benchmark_vpinn.py --device '0,1,2,3,4,5,6,7' --iter 200 ..."
  python benchmark_vpinn.py --device '0,1,2,4,5,6,7' --plotevery 1000 --iter 20000 --case 'PoissonInv, HeatInv'
fi

if [ "$1" != "c" ] && [ "$1" != "l" ] && [ ! -z "$1" ]; then
  echo "Invalid parameter! Use 'c' to clean up, 'l' to run the script, or both without arguments."
fi
