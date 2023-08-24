from typing import List
from collections import OrderedDict
import os
import jieba
from arguments import DataArguments, TrainingArguments
from log import TrainLog
from tqdm import tqdm

import torch


class Tokenizer:
    def __init__(self, data_args: DataArguments, training_args: TrainingArguments, train_log: TrainLog):
        self.pad_token = data_args.pad_token
        self.unk_token = data_args.unk_token
        self.bos_token = data_args.bos_token
        self.eos_token = data_args.eos_token
        self.training_args = training_args
        self.train_log = train_log
        self.vocab = OrderedDict()
        self.ids_to_tokens = OrderedDict()
        if self.training_args.distributed and self.training_args.local_rank not in [-1, 0]:
            torch.distributed.barrier()

        if os.path.exists(data_args.vocab_file):
            self.load_vocab(data_args.vocab_file)
        else:
            self.build_vocab(data_args)
        if self.train_log.enable:
            self.train_log.logger.info(
                f"vocab size: {self.vocab_size}")
            self.train_log.logger.info(
                f"pad_token_idx: {self.convert_token_to_id(self.pad_token)}")
            self.train_log.logger.info(
                f"unk_token_idx: {self.convert_token_to_id(self.unk_token)}")
            self.train_log.logger.info(
                f"bos_token_idx: {self.convert_token_to_id(self.bos_token)}")
            self.train_log.logger.info(
                f"eos_token_idx: {self.convert_token_to_id(self.eos_token)}")
            
        if self.training_args.distributed and  self.training_args.local_rank == 0:
            torch.distributed.barrier()

    def load_vocab(self, vocab_file: str):
        """
        Load vocab from vocab file.
        """
        if self.train_log.enable:
            self.train_log.logger.info(f"load vocab from {vocab_file}")

        with open(vocab_file, 'r', encoding='utf-8') as f:
            for index, line in enumerate(f.readlines()):
                self.vocab[line.replace("\n", "")] = index
                self.ids_to_tokens[index] = line.replace("\n", "")

    def build_vocab(self, data_args: DataArguments):
        """
        Build vocab from train corpus.
        """
        if self.train_log.enable:
            self.train_log.logger.info(f"build vocab from {data_args.train_corpus}")

        def init(tokens):
            """
            Initialize tokens with special tokens.
            """
            for token in [data_args.pad_token, data_args.unk_token, data_args.bos_token, data_args.eos_token]:
                tokens.append((token, 1))

            return tokens
        all_tokens = {}
        # Build vocab and ids_to_tokens
        with open(data_args.train_corpus, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for i, line in enumerate(tqdm(lines, total=len(lines), desc="build vocab")):
                line_tokens = list(line)
                for token in line_tokens:
                    if token in [" ", "\n", "\t"]:
                        continue
                    if token not in all_tokens.keys():
                        all_tokens[token] = 1
                    else:
                        all_tokens[token] += 1

        self.token_list = sorted(
            all_tokens.items(), key=lambda x: x[1], reverse=True)
        token_list = init([])
        token_list += [(token, freq) for (token, freq)
                       in self.token_list if freq >= data_args.min_freq]
        for index, (token, _) in enumerate(token_list):
            self.ids_to_tokens[index] = token
            self.vocab[token] = index

        with open(data_args.vocab_file, 'w+', encoding='utf-8') as f:
            if self.train_log.enable:
                self.train_log.logger.info("write vocab to {}".format(data_args.vocab_file))
            for index, line in enumerate(self.vocab):
                if index == len(self.vocab) - 1:
                    f.write(line)
                else:
                    f.write(line + "\n")

    def tokenize(self, text: str) -> List[str]:
        """Tokenize given text.
        """
        return list(text)

    def convert_token_to_id(self, token: str) -> int:
        """Convert a token (str) in an id (integer) using the vocab.
        """
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def convert_id_to_token(self, id: int) -> str:
        """Convert an id (integer) in a token (str) using the vocab.
        """
        return self.ids_to_tokens.get(id, self.unk_token)

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """Convert list of tokens in list of ids using the vocab.
        """
        return [self.convert_token_to_id(token) for token in tokens]

    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        """Convert list of ids in list of tokens using the vocab.
        """
        return [self.convert_id_to_token(id) for id in ids]

    @property
    def vocab_size(self) -> int:
        """Vocabulary size.
        """
        return len(self.vocab)

    @property
    def pad_token_id(self) -> int:
        """Id of pad_token in the vocab.
        """
        return self.convert_token_to_id(self.pad_token)

    @property
    def unk_token_id(self) -> int:
        """Id of unk_token in the vocab.
        """
        return self.convert_token_to_id(self.unk_token)

    @property
    def bos_token_id(self) -> int:
        """Id of bos_token in the vocab.
        """
        return self.convert_token_to_id(self.bos_token)

    @property
    def eos_token_id(self) -> int:
        """Id of eos_token in the vocab.
        """
        return self.convert_token_to_id(self.eos_token)


class PretrainedTokenizer(Tokenizer):
    def __init__(self, data_args: DataArguments, training_args: TrainingArguments, logger: TrainLog):

        super(PretrainedTokenizer, self).__init__(
            data_args, training_args, logger)

    def detokenize(self, tokens: List[str]) -> str:
        """Detokenize given tokens.
        """
        return self.tokenize(tokens)
