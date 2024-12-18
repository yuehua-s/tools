#!/bin/bash
# 将 input_file.jsonl 中随机取 1 万行到 output_file.jsonl 中
shuf -n 10000 input_file.jsonl > output_file.jsonl