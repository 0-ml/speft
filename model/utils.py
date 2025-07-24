import copy
import logging
import os
from dataclasses import dataclass
from functools import partial
from itertools import chain
from typing import Dict, Sequence

import datasets as hf_datasets
import torch
import transformers
from huggingface_hub import snapshot_download, hf_hub_download
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

# from ..utils import add_dataset_info


# logger = logging.getLogger(__name__)


def _load_cached_dataset(path, name=None, keep_in_memory=None, **kwargs):
    cache_dir = hf_datasets.config.HF_DATASETS_CACHE
    fpath = path.replace('/', '___')
    cache_dir = cache_dir / f"{fpath}/{name}/manual_save"
    try:
        dataset_dict = hf_datasets.load_dataset(
            path, name, keep_in_memory=keep_in_memory, **kwargs)
        if not cache_dir.exists():
            dataset_dict.save_to_disk(str(cache_dir))
    except ConnectionError:
        dataset_dict = hf_datasets.load_from_disk(
            cache_dir, keep_in_memory=keep_in_memory)
    return dataset_dict

