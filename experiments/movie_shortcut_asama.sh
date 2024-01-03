#!/bin/sh
python -W ignore making_movie.py --scenario $1 --load ./policy/${1}${2}/_${1}${2}0 --exp-name $1$2 --adv-policy ddpg --movie-fname ./results/${1}${2}/${1}${2}0.mp4