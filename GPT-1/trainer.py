import time
import json
from pathlib import Path
from tqdm import tqdm
import warnings
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from transformers.modelcard import TrainingSummary
from transformers.modeling_utils import PreTrainedModel, load_sharded_checkpoint, unwrap_model
from arguments import DataArguments, ModelArguments, TrainingArguments
from model import GPT2, GPT2LMHead
from tokenization import PretrainedTokenizer
from log import TrainLog
import copy


def timeit(method):
    def timed(*args, **kw):
        _args = args[0].training_args
        train_log = args[0].train_log

        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        if _args.distributed:
            if train_log.enable:
                train_log.logger.info('Function Time: {}\t>\t{:.0f} min {:.0f} sec'.format(
                    method.__name__, (te-ts)//60, (te-ts) % 60), local_rank=_args.local_rank)
        else:
            train_log.logger.info('Function Time: {}\t>\t{:.0f} min {:.0f} sec'.format(
                method.__name__, (te-ts)//60, (te-ts) % 60), local_rank=_args.local_rank)

        return result
    return timed


class Trainer:
    def __init__(self,
                 model: Union[PreTrainedModel, nn.Module] = None,
                 args: Union[ModelArguments, DataArguments,
                             TrainingArguments] = [],
                 data_collator: Optional[DataCollator] = None,
                 train_loader: Optional[DataLoader] = None,
                 test_loader: Optional[DataLoader] = None,
                 tokenizer: Optional[PretrainedTokenizer] = None,
                 optimizers: Tuple[torch.optim.Optimizer,
                                   torch.optim.lr_scheduler.LambdaLR] = (None, None),
                 train_log: Optional[TrainLog] = None):
        [self.model_args, self.data_args, self.training_args] = args
        self.device = self.training_args.device
        if self.training_args.do_pretrain:
            self.model = GPT2LMHead(model)
            self.model.to(self.device)

        self.tokenizer = tokenizer
        [self.optimizer, self.lr_scheduler] = optimizers
        self.data_collator = data_collator
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.do_pretrain = self.training_args.do_pretrain
        self.do_eval = self.training_args.do_eval
        self.do_finetune = self.training_args.do_finetune
        self.do_predict = self.training_args.do_predict
        self.n_gpus = self.training_args.n_gpus
        self.device = torch.device(
            self.training_args.device, self.training_args.local_rank) if self.training_args.distributed else torch.device(self.training_args.device) 
        self.distributed = self.training_args.distributed
        self.vocab_size = self.tokenizer.vocab_size

        self.train_log = train_log
        
        
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.tokenizer.pad_token_id).to(self.device)

        if self.distributed:
            if self.training_args.local_rank in [-1, 0]:
                self.writer = SummaryWriter(log_dir=self.training_args.train_dir)

            self.model = DistributedDataParallel(
                self.model, device_ids=[
                    self.training_args.local_rank], output_device=self.training_args.local_rank, find_unused_parameters=True
            )
        else:
            self.writer = SummaryWriter(log_dir=self.training_args.train_dir)
        if self.train_log.enable:
            self.train_log.logger.info(f"The Pretrain model is: {self.model}")
            self.train_log.logger.info(f"Ready to train on {self.device}")

    @timeit
    def train(self, epoch):
        if self.do_pretrain:
            if self.train_log.enable:
                self.train_log.logger.info(f"Start Pretrain on epoch {epoch}")
            self.pretrain(epoch)
        elif self.do_eval:
            self.evaluate(epoch)
        elif self.do_predict:
            self.predict(epoch)
        elif self.do_finetune:
            self.finetune(epoch)

    def pretrain(self, epoch):
        losses = 0
        batch_size, smaples = len(self.train_loader), len(
            self.train_loader.dataset)
        # 开启训练模式
        self.model.train()

        for step, batch in enumerate(tqdm(self.train_loader, desc="Iteration")):
            inputs = batch[0].to(self.device)
            targets = inputs[:, 1:].contiguous()
            if step < 1:
                if self.train_log.enable:
                    input = copy.deepcopy(batch[0])
                    self.train_log.logger.info(f"Input: {self.tokenizer.convert_ids_to_tokens(input[0].tolist())} \nTarget: {self.tokenizer.convert_ids_to_tokens(input[0, 1:].tolist())}")
            lm_logist = self.model(inputs)

            lm_logist = lm_logist[:, :-1].contiguous()

            loss = self.criterion(
                lm_logist.view(-1, self.vocab_size), targets.view(-1))

            losses += loss.item()

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

            self.lr_scheduler.step()

            if self.training_args.local_rank in [-1, 0]:
                self.writer.add_scalar(
                    "loss-pretrain", loss.item(), epoch * batch_size + step)
                self.writer.add_scalar(
                    "lr", self.optimizer.param_groups[0]['lr'], epoch * batch_size + step)

                if step % self.training_args.logging_steps == 0 and step != 0:
                    if self.train_log.enable:
                        self.train_log.logger.info("Iteration {} ({}/{})\tLoss: {:.4f}\tLearningRate: {:.6f}".format(step,
                                                                                                               step, batch_size, losses / step, self.optimizer.param_groups[0]['lr']))

        if self.train_log.enable:
            self.train_log.logger.info(
                "Epoch {} done\tLoss: {:.4f}".format(epoch, losses / batch_size))

    def evaluate(self, epoch):
        losses, accuracies = 0.0, 0.0
        batch_size, smaples = len(self.test_loader), len(
            self.test_loader.dataset)
        self.model.eval()

        with torch.no_grad():
            for step, batch in enumerate(tqdm(self.test_loader, desc="Iteration")):
                if self.training_args.do_pretrain:
                    inputs = batch[0].to(self.device)
                    targets = inputs[:, 1:].contiguous()
                    lm_logist = self.model(inputs)

                    lm_logist = lm_logist[:, :-1].contiguous()

                    loss = self.criterion(
                        lm_logist.view(-1, self.vocab_size), targets.view(-1))

                    losses += loss.item()

                    if self.training_args.local_rank in [-1, 0]:
                        self.writer.add_scalar(
                            "loss-evaluate", loss.item(), epoch * batch_size + step)
                        self.writer.add_scalar(
                            "lr", self.optimizer.param_groups[0]['lr'], epoch * batch_size + step)

                elif self.training_args.do_finetune:
                    inputs, labels = map(lambda x: x.to(self.device), batch)

                    lm_logits, cls_logits = self.model(inputs)
                    lm_logits = lm_logits[:, :-1].contiguous()

                    lm_loss = self.criterion(
                        lm_logits.view(-1, self.vocab_size), inputs[:, 1:].contiguous().view(-1))
                    cls_loss = self.cls_criterion(cls_logits, labels)
                    loss = cls_loss + \
                        (self.training_args.auxiliary_ratio * lm_loss)

                    losses += loss.item()
                    acc = (cls_logits.argmax(dim=-1) ==
                           labels).to(dtype=cls_logits.dtype).mean()
                    accuracies += acc

                    if self.training_args.local_rank in [-1, 0]:
                        self.writer.add_scalar(
                            'Loss/fine-tune(eval)', loss.item(), ((epoch-1)*batch_size)+step)
                        self.writer.add_scalar(
                            'Accuracy/fine-tune(eval)', acc, ((epoch-1)*batch_size)+step)

        if self.train_log.enable:
            self.train_log.logger.info('Eval Epoch {} [rank: {}]\t>\tLoss: {:.4f} / Acc: {:.1f}%'.format(
                epoch, self.training_args.local_rank, losses/batch_size, accuracies/batch_size*100.))

    def predict(self, epoch):
        pass

    def finetune(self, epoch):

        pass

    def save(self, epoch, model_prefix='model', root='.model', model_suffix='.ckpt'):
        path = Path(root) / (model_prefix + model_suffix)
        if self.train_log.enable:
            self.train_log.logger.info("Saving model to %s" % path)
        if self.training_args.local_rank == 0:
            if not path.parent.exists():
                path.parent.mkdir()

        if self.training_args.distributed:
            if self.training_args.local_rank == 0:
                torch.save(self.model, path)
        else:
            torch.save(self.model, path)

    def save_parameter(self, file):
        total_num = sum(p.numel() for p in self.model.parameters())
        trainable_num = sum(p.numel()
                            for p in self.model.parameters() if p.requires_grad)
        with open(file, 'w+', encoding='utf-8') as f:
            f.write(
                f"Parameter number: Total: {total_num}, Trainable: {trainable_num}")
            f.write("\n")
            for name, param in self.model.named_parameters():
                f.write(f"{name}\t{param.size()}\n")

            f.write("\n")

            f.write("Model Structure:\n")

            f.write(str(self.model))
