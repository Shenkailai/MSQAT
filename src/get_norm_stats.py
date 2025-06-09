"""Utility script to compute dataset normalization statistics."""

import torch
import numpy as np

from dataloader import NisqaDataset


def compute_stats(datapath, label_csv, audio_conf):
    """Return mean and std over the given dataset."""
    loader = torch.utils.data.DataLoader(
        NisqaDataset(datapath, label_csv=label_csv, audio_conf=audio_conf, is_train=False),
        batch_size=1000, shuffle=False, num_workers=0, pin_memory=True,
    )
    mean, std = [], []
    for audio_input, _ in loader:
        mean.append(torch.mean(audio_input))
        std.append(torch.std(audio_input))
    return float(torch.mean(torch.stack(mean))), float(torch.mean(torch.stack(std)))


if __name__ == "__main__":
    audio_conf = {
        "num_mel_bins": 128,
        "target_length": 1024,
        "freqm": 0,
        "timem": 0,
        "mixup": 0,
        "skip_norm": True,
        "mode": "train",
        "dataset": "nisqa",
        "padding_mode": "repetitive",
    }
    mean, std = compute_stats(
        r"E:\SQA\TencentCorups",
        r"E:\SQA\TencentCorups\without.csv",
        audio_conf,
    )
    print(mean, std)
