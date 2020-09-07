import argparse
import glob
import os
import json
import time
import logging
import random
import re
from itertools import chain
from string import punctuation

import nltk
from nltk.tokenize import sent_tokenize

import pandas as pd
import numpy as np
import torch
import pytorch_lightning as pl

from utils import set_seed
from models import T5FineTuner
from transformers import T5Tokenizer

from torch.utils.data import DataLoader, random_split

logger = logging.getLogger(__name__)

set_seed(42)


args_dict = dict(
    data_dir="",  # path for data files
    output_dir="",  # path to save the checkpoints
    model_name_or_path="t5-base",
    tokenizer_name_or_path="t5-base",
    max_seq_length=48,
    learning_rate=1e-4,
    weight_decay=0.0,
    adam_epsilon=1e-8,
    warmup_steps=0,
    train_batch_size=48,
    eval_batch_size=48,
    num_train_epochs=2,
    gradient_accumulation_steps=4,
    n_gpu=1,
    early_stop_callback=False,
    fp_16=False,  # if you want to enable 16-bit training then install apex and set this to true
    opt_level="O1",  # you can find out more on optimisation levels here https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
    max_grad_norm=1.0,  # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
    seed=42,
)

args_dict.update({"data_dir": "data", "output_dir": "com-gec", "num_train_epochs": 2})
args = argparse.Namespace(**args_dict)

checkpoint_callback = pl.callbacks.ModelCheckpoint(
    filepath=args.output_dir,
    prefix="com_base_",
    monitor="val_loss",
    save_top_k=-1,
    period=-1,
)

tokenizer = T5Tokenizer.from_pretrained(args.tokenizer_name_or_path)

train_params = dict(
    accumulate_grad_batches=args.gradient_accumulation_steps,
    gpus=args.n_gpu,
    max_epochs=args.num_train_epochs,
    early_stop_callback=False,
    precision=16 if args.fp_16 else 32,
    amp_level=args.opt_level,
    gradient_clip_val=args.max_grad_norm,
    checkpoint_callback=checkpoint_callback,
    val_check_interval=0.1,
)

if __name__ == "__main__":
    model = T5FineTuner(args)
    trainer = pl.Trainer(**train_params)
    trainer.fit(model)
