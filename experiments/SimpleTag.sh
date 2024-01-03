#!/bin/sh
CUDA_VISIBLE_DEVICES=-1 python simple_train.py --scenario simple_tag --save-rate 100 --num-episodes 1000 --save-dir ./policy/SimpleTag/ --exp-name SimpleTag --without-curriculum