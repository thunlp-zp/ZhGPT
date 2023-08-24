from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
time_now = datetime.now()
time_str = time_now.strftime('%Y%m%d%H')


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    checkpoint_dir: Optional[str] = field(
        default="pth",
        metadata={
            "help": "Path to checkpoint directory."
            }
    )

    seq_len: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        }
    )
    d_model: Optional[int] = field(
        default=768,
        metadata={
            "help": (
                "The dimensionality of the model's hidden states."
            )
        }
    )
    n_layers: Optional[int] = field(
        default=12,
        metadata={
            "help": (
                "The number of decoder layers."
            )
        }
    )
    n_heads: Optional[int] = field(
        default=12,
        metadata={
            "help": (
                "The number of attention heads."
            )
        }
    )
    d_ff: Optional[int] = field(
        default=3072,
        metadata={
            "help": (
                "The dimensionality of the feedforward network model."
            )
        }
    )
    dropout: Optional[float] = field(
        default=0.1,
        metadata={
            "help": (
                "The attention dropout probability."
            )
        }
    )
    residual_dropout: Optional[float] = field(
        default=0.1,
        metadata={
            "help": (
                "The residual dropout probability."
            )
        }
    )
    embed_dropout: Optional[float] = field(
        default=0.5,
        metadata={
            "help": (
                "The embedding dropout probability."
            )
        }
    )
    model_prefix: Optional[str] = field(
        default="GPT2",
        metadata={
            "help": (
                "The model prefix."
            )
        }
    )
    model_suffix: Optional[str] = field(
        default=".ckpt",
        metadata={
            "help": (
                "The model suffix."
            )
        }
    )
    max_seq_len: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded. The context window of GPT-2 is 512."
            )
        }
    )
   


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    vocab_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "The vocab file path."
        }
    )
    train_corpus: Optional[str] = field(
        default=None,
        metadata={
            "help": "The train corpus path."
        }
    )
    test_corpus: Optional[str] = field(
        default=None,
        metadata={
            "help": "The test corpus path."
        }
    )
    pad_token: Optional[str] = field(
        default='<pad>',
        metadata={
            "help": "The pad token."
        }
    )
    unk_token: Optional[str] = field(
        default='<unk>',
        metadata={
            "help": "The unk token."
        }
    )
    bos_token: Optional[str] = field(
        default='<bos>',
        metadata={
            "help": "The bos token."
        }
    )
    eos_token: Optional[str] = field(
        default='<eos>',
        metadata={
            "help": "The eos token."
        }
    )
    min_freq: Optional[int] = field(
        default=1,
        metadata={
            "help": "The min freq."
        }
    )


@dataclass
class TrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    signature: Optional[str] = field(
        default=time_str,
        metadata={
            "help": "Train signature."
        }
    )

    train_dir: Optional[str] = field(
        default="/home/zhoupeng/zh2nlp_expr/ZhGPT/GPT-2/train_log",
        metadata={
            "help": "Path to train directory."
            }
    )

    batch_size: Optional[int] = field(
        default=34,
        metadata={
            "help": "Batch size."
        }
    )
    # same as the paper of GPT-2
    learning_rate: Optional[float] = field(
        default=2.5e-4,
        metadata={
            "help": (
                "The initial learning rate for Adam."
            )
        }
    )
    epoches: Optional[int] = field(
        default=50,
        metadata={
            "help": "Epoches."
        }
    )
    n_workers: Optional[int] = field(
        default=0,
        metadata={
            "help": "Number of workers."
        }
    )
    cuda_visible_devices: Optional[str] = field(
        default="0, 1, 2, 3, 4, 5, 6, 7",
        metadata={
            "help": "Cuda visible devices."
        }
    )
    do_eval: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to evaluate the model."
        }
    )
    do_finetune: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to fine-tune the model."
        }
    )
    do_pretrain: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Whether to train the model."
        }
    )
    do_predict: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to predict the model."
        }
    )
    do_save: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Whether to save the model."
        }
    )
    do_save_steps: Optional[int] = field(
        default=1000,
        metadata={
            "help": "The steps to save the model."
        }
    )
    save_interval_epoch: Optional[int] = field(
        default=1,
        metadata={
            "help": "The interval epoch to save the model."
        }
    )
    n_gpus: Optional[int] = field(
        default=4,
        metadata={
            "help": "The number of GPUs."
        })
    distributed: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to use distributed training. Default is False."
        })
    device: Optional[str] = field(
        default='cuda',
        metadata={
            "help": "The device to use."
        })
    seed: Optional[int] = field(
        default=42,
        metadata={
            "help": "The seed to use."
        })
    log_dir: Optional[str] = field(
        default='/home/zhoupeng/zh2nlp_expr/ZhGPT/GPT-2/logs',
        metadata={
            "help": "The log directory."
        })
    local_rank: Optional[int] = field(
        default=0,
        metadata={
            "help": "The local rank."
        })
    fp16: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to use fp16."
        }
    )
    should_log: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Whether to use logger."
        })
    mode: Optional[str] = field(
        default='Train',
        metadata={
            "help": "The mode."
        })
    backend: Optional[str] = field(
        default='nccl',
        metadata={
            "help": "The backend."
        })
    master_addr: Optional[str] = field(
        default='localhost',
        metadata={
            "help": "The master address."
        })
    master_port: Optional[str] = field(
        default="12355",
        metadata={
            "help": "The master port."
        })
    master_rank: Optional[str] = field(
        default="0",
        metadata={
            "help": "The master rank."
        })
    init_distributed_method: Optional[str] = field(
        default='env://',
        metadata={
            "help": "The init distributed method."
        })
    logging_steps: Optional[int] = field(
        default=500,
        metadata={
            "help": "The logging steps."
        })
    

