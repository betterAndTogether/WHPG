#!/usr/bin/env bash


# wikiHow
#CUDA_VISIBLE_DEVICES=3 python trainer.py --max_seq_len 80 --num_epochs 120 --window_size 0 --model bert-base-uncased --datafile data/WHPG/ > ./log/ALL_wikiHow_log.txt
#CUDA_VISIBLE_DEVICES=3 python trainer.py --max_seq_len 80 --num_epochs 120 --window_size 5 --model bert-base-uncased --datafile data/WHPG/ > ./log/5_wikiHow_log.txt
#CUDA_VISIBLE_DEVICES=3 python trainer.py --max_seq_len 80 --num_epochs 120 --window_size 10 --model bert-base-uncased --datafile data/WHPG/ > ./log/10_wikiHow_log.txt
#CUDA_VISIBLE_DEVICES=3 python trainer.py --max_seq_len 80 --num_epochs 120 --window_size 20 --model bert-base-uncased --datafile data/WHPG/ > ./log/20_wikiHow_log.txt

# cooking domain
#CUDA_VISIBLE_DEVICES=4 python trainer.py --max_seq_len 80 --num_epochs 120 --window_size 0 --model bert-base-uncased --datafile data/cooking/ > ./log/ALL_Cooking_log.txt
#CUDA_VISIBLE_DEVICES=4 python trainer.py --max_seq_len 80 --num_epochs 120 --window_size 5 --model bert-base-uncased --datafile data/cooking/ > ./log/5_Cooking_log.txt
#CUDA_VISIBLE_DEVICES=4 python trainer.py --max_seq_len 80 --num_epochs 120 --window_size 10 --model bert-base-uncased --datafile data/cooking/ > ./log/10_Cooking_log.txt
#CUDA_VISIBLE_DEVICES=4 python trainer.py --max_seq_len 80 --num_epochs 120 --window_size 20 --model bert-base-uncased --datafile data/cooking/ > ./log/20_Cooking_log.txt

# MAM domain
CUDA_VISIBLE_DEVICES=5 python trainer.py --max_seq_len 80 --num_epochs 120 --window_size 0 --model bert-base-uncased --datafile data/mam/ > ./log/ALL_MAM_log.txt
CUDA_VISIBLE_DEVICES=5 python trainer.py --max_seq_len 80 --num_epochs 120 --window_size 5 --model bert-base-uncased --datafile data/mam/ > ./log/5_MAM_log.txt
CUDA_VISIBLE_DEVICES=5 python trainer.py --max_seq_len 80 --num_epochs 120 --window_size 10 --model bert-base-uncased --datafile data/mam/ > ./log/10_MAM_log.txt
CUDA_VISIBLE_DEVICES=5 python trainer.py --max_seq_len 80 --num_epochs 120 --window_size 20 --model bert-base-uncased --datafile data/mam/ > ./log/20_MAM_log.txt
