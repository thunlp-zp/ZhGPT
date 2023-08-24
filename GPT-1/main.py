
import sys
import os
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, DistributedSampler
from datasets import load_dataset
from model import GPT1
from tokenization import PretrainedTokenizer
import transformers
from transformers import (
    HfArgumentParser,
    set_seed,
)

import torch.distributed as dist
from arguments import DataArguments, ModelArguments, TrainingArguments
from trainer import Trainer
from data_util import create_examples
from log import TrainLog


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    print("training_args.local_rank:", training_args.local_rank)
    if training_args.distributed:
        # torch.distributed.lanch will set the following environment variables automatically, so we don't need to set them manually:
        # os.environ['MASTER_ADDR'] = training_args.master_addr
        # os.environ['MASTER_PORT'] = training_args.master_port
        # os.environ['RANK'] = training_args.master_rank
        # os.environ['WORLD_SIZE'] = "1"
        os.environ["CUDA_VISIBLE_DEVICES"] = training_args.cuda_visible_devices
        # set gpu device for each process
        torch.cuda.set_device(training_args.local_rank)
        # we will use the nccl backend for distributed training, it supports cross-machine training.
        dist.init_process_group(backend=training_args.backend)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Initialize the log
    train_log = TrainLog(
        mode=training_args.mode, training_args=training_args, is_master=dist.get_rank() in [-1, 0] if training_args.distributed else True)
    if train_log.enable:
        train_log.logger.info(f"train_dir:{training_args.train_dir}")
    
    # TODO: for sync training_args.train_dir
    if training_args.distributed and training_args.local_rank not in [-1, 0]:
        dist.barrier()
    if training_args.distributed and training_args.local_rank == 0:
        dist.barrier()
    # train_log is enabled only for master process
    if train_log.enable:
        train_log.logger.info(
            f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpus: {training_args.n_gpus}, "
            + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}")
        train_log.logger.info(f"Model parameters {model_args}")
        train_log.logger.info(f"Data parameters {data_args}")
        train_log.logger.info(
            f"Training/evaluation parameters {training_args}")

    # Initialize the tokenizer
    tokenizer = PretrainedTokenizer(data_args, training_args, train_log)

    if train_log.enable:
        train_log.logger.info(f"Loading Data")

    train_dataset = create_examples(
        [model_args, data_args, training_args], tokenizer, train_log, mode='train')

    train_sampler = DistributedSampler(
        train_dataset) if training_args.distributed else RandomSampler(train_dataset)
    train_loader = DataLoader(train_dataset, sampler=train_sampler,
                              batch_size=training_args.batch_size, num_workers=training_args.n_workers)
    test_loader = None
    if training_args.do_eval:
        test_dataset = create_examples(
            [model_args, data_args, training_args], tokenizer, train_log, mode='test')
        test_sampler = DistributedSampler(
            test_dataset) if training_args.distributed else RandomSampler(test_dataset)
        test_loader = DataLoader(test_dataset, sampler=test_sampler,
                                 batch_size=training_args.batch_size, num_workers=training_args.n_workers)
    # Load dataset
    gpt1 = GPT1(tokenizer.vocab_size,
                model_args.d_model,
                model_args.seq_len,
                model_args.n_layers,
                model_args.n_heads,
                model_args.d_ff,
                model_args.dropout,
                model_args.residual_dropout,
                model_args.embed_dropout,
                tokenizer.pad_token_id)

    optimizer = torch.optim.Adam(
        gpt1.parameters(), lr=training_args.learning_rate)

    # use cosine annealing lr scheduler for training GPT-1 as the paper suggested, T_max is the number of epoches
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=training_args.epoches, eta_min=0)
    # TODO: add LR scheduler
    trainer = Trainer(model=gpt1,
                      args=[model_args, data_args, training_args],
                      train_loader=train_loader,
                      test_loader=test_loader,
                      tokenizer=tokenizer,
                      optimizers=[optimizer, scheduler],
                      train_log=train_log)

    for epoch in range(1, training_args.epoches + 1):
        trainer.train(epoch)
        if training_args.do_save and epoch % training_args.save_interval_epoch == 0:
            trainer.save(epoch, model_prefix=model_args.model_prefix,
                         root=os.path.join(training_args.train_dir, model_args.checkpoint_dir), model_suffix=model_args.model_suffix)

        if training_args.do_finetune:
            trainer.evaluate(epoch)

    trainer.save_parameter(os.path.join(
        training_args.train_dir, "GPT1_parameter.txt"))

    # release logger
    if training_args.should_log:
        train_log.remove()


if __name__ == "__main__":
    main()
