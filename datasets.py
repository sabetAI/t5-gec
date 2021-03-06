import linecache
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Union

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, Sampler

from transformers import BartTokenizer

from tqdm import tqdm
import os


class SortishSampler(Sampler):
    "Go through the text data by order of src length with a bit of randomness. From fastai repo."

    def __init__(self, data, batch_size):
        self.data, self.bs = data, batch_size

    def key(self, i):
        return self.data[i]

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self):
        idxs = np.random.permutation(len(self.data))
        sz = self.bs * 50
        ck_idx = [idxs[i : i + sz] for i in range(0, len(idxs), sz)]
        sort_idx = np.concatenate(
            [sorted(s, key=self.key, reverse=True) for s in ck_idx]
        )
        sz = self.bs
        ck_idx = [sort_idx[i : i + sz] for i in range(0, len(sort_idx), sz)]
        max_ck = np.argmax(
            [self.key(ck[0]) for ck in ck_idx]
        )  # find the chunk with the largest key,
        ck_idx[0], ck_idx[max_ck] = (
            ck_idx[max_ck],
            ck_idx[0],
        )  # then make sure it goes first.
        sort_idx = (
            np.concatenate(np.random.permutation(ck_idx[1:]))
            if len(ck_idx) > 1
            else np.array([], dtype=np.int)
        )
        sort_idx = np.concatenate((ck_idx[0], sort_idx))
        return iter(sort_idx)


class AbstractSeq2SeqDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_dir,
        max_source_length,
        max_target_length,
        type_path="train",
        n_obs=None,
        src_lang=None,
        tgt_lang=None,
        prefix="",
    ):
        super().__init__()
        self.src_file = Path(data_dir).joinpath(type_path + ".src")
        self.tgt_file = Path(data_dir).joinpath(type_path + ".trg")
        self.src_lens = self.get_char_lens(self.src_file)
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        assert min(self.src_lens) > 0, f"found empty line in {self.src_file}"
        self.tokenizer = tokenizer
        self.prefix = prefix
        if n_obs is not None:
            self.src_lens = self.src_lens[:n_obs]
        self.pad_token_id = self.tokenizer.pad_token_id
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.add_prefix_space = isinstance(self.tokenizer, BartTokenizer)

    def __len__(self):
        return len(self.src_lens)

    @staticmethod
    def get_char_lens(data_file):
        return [len(x) for x in Path(data_file).open().readlines()]

    def make_sortish_sampler(self, batch_size):
        return SortishSampler(self.src_lens, batch_size)

    def __getitem__(self, item):
        raise NotImplementedError("You must implement this")

    def collate_fn(self, batch):
        raise NotImplementedError("You must implement this")


class Seq2SeqDataset(AbstractSeq2SeqDataset):
    """A dataset that calls prepare_seq2seq_batch."""

    def __getitem__(self, index) -> Dict[str, str]:
        index = index + 1  # linecache starts at 1
        source_line = self.prefix + linecache.getline(str(self.src_file), index).rstrip(
            "\n"
        )
        tgt_line = linecache.getline(str(self.tgt_file), index).rstrip("\n")
        assert source_line, f"empty source line for index {index}"
        assert tgt_line, f"empty tgt line for index {index}"
        return {
            "tgt_texts": tgt_line,
            "src_texts": source_line,
        }

    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        """Call prepare_seq2seq_batch."""
        batch_encoding = self.tokenizer.prepare_seq2seq_batch(
            [x["src_texts"] for x in batch],
            tgt_texts=[x["tgt_texts"] for x in batch],
            max_length=self.max_source_length,
            max_target_length=self.max_target_length,
            return_tensors="pt",
        )
        return batch_encoding


class LegacySeq2SeqDataset(AbstractSeq2SeqDataset):
    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        """Call tokenizer on src and tgt_lines"""
        index = index + 1  # linecache starts at 1
        source_line = self.prefix + linecache.getline(str(self.src_file), index).rstrip(
            "\n"
        )
        tgt_line = linecache.getline(str(self.tgt_file), index).rstrip("\n")
        assert source_line, f"empty source line for index {index}"
        assert tgt_line, f"empty tgt line for index {index}"
        source_inputs = encode_line(self.tokenizer, source_line, self.max_source_length)
        target_inputs = encode_line(self.tokenizer, tgt_line, self.max_target_length)

        source_ids = source_inputs["input_ids"].squeeze()
        target_ids = target_inputs["input_ids"].squeeze()
        src_mask = source_inputs["attention_mask"].squeeze()
        return {
            "input_ids": source_ids,
            "attention_mask": src_mask,
            "labels": target_ids,
        }

    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        input_ids = torch.stack([x["input_ids"] for x in batch])
        masks = torch.stack([x["attention_mask"] for x in batch])
        target_ids = torch.stack([x["labels"] for x in batch])
        pad_token_id = self.pad_token_id
        y = trim_batch(target_ids, pad_token_id)
        source_ids, source_mask = trim_batch(
            input_ids, pad_token_id, attention_mask=masks
        )
        batch = {
            "input_ids": source_ids,
            "attention_mask": source_mask,
            "labels": y,
        }
        return batch


class Seq2SeqIterDataset(IterableDataset):
    def __init__(
        self,
        tokenizer,
        data_dir,
        max_source_length,
        max_target_length,
        world_size,
        rank,
        type_path="train",
    ):
        super().__init__()
        self.type_path = type_path
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.pad_token_id = self.tokenizer.pad_token_id
        self.world_size = world_size
        self.rank = rank

    def __len__(self):
        def file_len(fname):
            with open(fname) as f:
                for i, l in enumerate(f):
                    pass
            return i + 1

        src_file_path = Path(self.data_dir).joinpath(self.type_path + ".src")
        self.len = file_len(src_file_path)
        return self.len

    def process_data(self, src_file, trg_file, n_skips):
        for _ in tqdm(range(n_skips)):
            next(src_file)
            next(trg_file)

        for src_line, trg_line in zip(src_file, trg_file):
            assert src_line, f"empty source line"
            assert trg_line, f"empty tgt line"
            src_inputs = encode_line(
                self.tokenizer, src_line.strip("\n"), self.max_source_length
            )
            trg_inputs = encode_line(
                self.tokenizer, trg_line.strip("\n"), self.max_target_length
            )
            src_ids = src_inputs["input_ids"].squeeze()
            print(src_ids)
            trg_ids = trg_inputs["input_ids"].squeeze()
            src_mask = src_inputs["attention_mask"].squeeze()
            yield {"input_ids": src_ids, "attention_mask": src_mask, "labels": trg_ids}

    def __iter__(self) -> Dict[str, torch.Tensor]:
        worker_info = torch.utils.data.get_worker_info()
        if self.world_size:
            chunk_size = self.len // (worker_info.num_workers * self.world_size)
        else:
            chunk_size = self.len // (worker_info.num_workers)
        n_skips = ((self.rank * worker_info.num_workers) + worker_info.id) * chunk_size
        self.src_file_path = Path(self.data_dir).joinpath(self.type_path + ".src")
        self.trg_file_path = Path(self.data_dir).joinpath(self.type_path + ".trg")
        self.src_file = open(self.src_file_path, encoding="utf-8")
        self.trg_file = open(self.trg_file_path, encoding="utf-8")

        return self.process_data(self.src_file, self.trg_file, n_skips)

    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        input_ids = torch.stack([x["input_ids"] for x in batch])
        masks = torch.stack([x["attention_mask"] for x in batch])
        target_ids = torch.stack([x["labels"] for x in batch])
        y = trim_batch(target_ids, self.pad_token_id)
        source_ids, source_mask = trim_batch(
            input_ids, self.pad_token_id, attention_mask=masks
        )
        batch = {
            "input_ids": source_ids,
            "attention_mask": source_mask,
            "labels": y,
        }
        return batch


def encode_line(
    tokenizer, line, max_length, pad_to_max_length=True, return_tensors="pt"
):
    # """Only used by LegacyDataset"""
    extra_kw = (
        {"add_prefix_space": True} if isinstance(tokenizer, BartTokenizer) else {}
    )
    return tokenizer(
        [line],
        max_length=max_length,
        padding="max_length" if pad_to_max_length else None,
        return_tensors=return_tensors,
        **extra_kw,
    )


def trim_batch(
    input_ids, pad_token_id, attention_mask=None,
):
    """Remove columns that are populated exclusively by pad_token_id"""
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask is None:
        return input_ids[:, keep_column_mask]
    else:
        return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask])
