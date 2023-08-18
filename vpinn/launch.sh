#!/bin/bash

# 检查参数
if [ "$1" == "c" ] || [ -z "$1" ]; then
  echo "清理 ./fig, ./log, ./model 目录下所有文件..."
  rm -rf ./fig/*
  rm -rf ./log/*
  rm -rf ./model/*
fi

if [ "$1" == "l" ] || [ -z "$1" ]; then
  # echo "运行 python benchmark_vpinn.py --device '0,1,2,3,4,5,6,7' --iter 200 ..."
  python benchmark_vpinn.py --device '0,1,2,4,5,6,7' --plotevery 20000 --iter 20000 --params True 
fi

if [ "$1" != "c" ] && [ "$1" != "l" ] && [ ! -z "$1" ]; then
  echo "无效参数! 请使用 'c' 来清理或 'l' 来运行脚本，或不带参数来执行两者。"
fi
