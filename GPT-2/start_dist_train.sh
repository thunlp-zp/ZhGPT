#!/bin/bash


/home/zhoupeng/miniconda3/bin/python3 -m torch.distributed.launch \
    --nproc_per_node 4 \
    --nnodes 1 \
    --node_rank=0 \
    /home/zhoupeng/zh2nlp_expr/ZhGPT/GPT-2/main.py \
    --train_corpus /home/zhoupeng/zh2nlp_expr/ZhGPT/GPT-2/data/poetry.txt \
    --vocab_file /home/zhoupeng/zh2nlp_expr/ZhGPT/GPT-2/data/poetry_vocab.txt \
    --do_pretrain \
    --distributed \
    --n_gpus 4
