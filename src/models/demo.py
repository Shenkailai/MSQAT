import os
import torch
import torch.nn as nn
import torchaudio
import numpy as np
def get_shape(fstride, tstride, input_fdim=128, input_tdim=1024):
    test_input = torch.randn(1, 1, input_fdim, input_tdim)
    test_proj = nn.Conv2d(1, 768, kernel_size=(16, 16), stride=(fstride, tstride))
    test_out = test_proj(test_input)
    f_dim = test_out.shape[2]
    t_dim = test_out.shape[3]
    return f_dim, t_dim


# f_dim, t_dim = get_shape(fstride=16, tstride=16)
# print(f_dim, t_dim)
# os.makedirs("./output/models/epoch1")

waveform, sr = torchaudio.load(r'E:\SQA\NISQA_Corpus\NISQA_TRAIN_SIM\deg\c01815_tsp_2_MK_20.wav')
fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                                                  window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10)

target_length = 1024
n_frames = fbank.shape[0]

dup_times = target_length // n_frames
remain = target_length - n_frames * dup_times
to_dup = [fbank for t in range(dup_times)]
pad = fbank[:remain, :]
to_dup.append(fbank[:remain, :])
fbank_padded = torch.Tensor(np.concatenate(to_dup, axis = 0))

