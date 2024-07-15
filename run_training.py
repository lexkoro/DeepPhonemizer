import logging.config
import random
from collections import defaultdict
from glob import glob
from pathlib import Path

import torch
import torch.multiprocessing as mp

from dp.preprocess import preprocess
from dp.train import train
from dp.utils.io import read_config

config_file_path = Path("logging.yaml")
config = read_config(config_file_path)
logging.config.dictConfig(config)


if __name__ == "__main__":
    tsv_files = glob("/home/DeepPhonemizer/tsv-filtered/*.tsv")
    train_data = []
    for f in tsv_files:
        language_code = f.split("/")[-1].split("_")[0]
        with open(
            f,
            "r",
            encoding="utf-8",
        ) as f:
            lines = f.readlines()
        # Prepare data as tuples (lang, word, phoneme)
        lines = [l.replace(" ", "").replace("\n", "") for l in lines]
        splits = [l.split("\t") for l in lines]
        for grapheme, phoneme in splits:
            if len(grapheme) > 0 and len(phoneme) > 0:
                train_data.append((language_code, grapheme, phoneme))

    # split into train and validation
    train_grouped_by_lang = defaultdict(list)
    validate_grouped_by_lang = defaultdict(list)

    for lang, word, phoneme in train_data:
        train_grouped_by_lang[lang].append((word, phoneme))

    for lang, data in train_grouped_by_lang.items():
        n = min(15, int(len(data) * 0.01))
        validate_grouped_by_lang[lang].extend(data[:n])
        train_grouped_by_lang[lang] = data[n:]

    train_data = []
    validation_data = []
    for lang, data in train_grouped_by_lang.items():
        for grapheme, phoneme in data:
            train_data.append((lang, grapheme, phoneme))

    for lang, data in validate_grouped_by_lang.items():
        for grapheme, phoneme in data:
            validation_data.append((lang, grapheme, phoneme))

    config_file = "dp/configs/forward_config.yaml"
    restore_path = (
        "/workspace/pretrained_models/g2p/forward_checkpoints_v3/latest_model.pt"
    )
    # restore_path = None

    preprocess(
        config_file=config_file,
        train_data=train_data,
        val_data=validation_data,
        deduplicate_train_data=True,
    )

    num_gpus = torch.cuda.device_count()

    # if num_gpus > 1:
    #     mp.spawn(
    #         train,
    #         nprocs=num_gpus,
    #         args=(num_gpus, config_file, restore_path),
    #     )
    # else:
    train(
        rank=0,
        num_gpus=1,
        config_file=config_file,
        checkpoint_file=restore_path,
    )
