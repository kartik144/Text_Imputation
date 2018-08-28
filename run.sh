#!/usr/bin/env bash

source activate torch
python3 main.py --model GRU --emsize 600 --nhid 600 --epochs 50 --cuda
