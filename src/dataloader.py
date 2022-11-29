import csv
import json
import math
import os
import torchaudio
import numpy as np
import torch
from torch import Tensor
import torch.nn.functional
from torch.utils.data import Dataset
import random

def make_index_dict(label_csv, isTrain, Eval):
    index_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            if Eval == None:
                if isTrain:
                    if 'train' in row['db'] or 'TRAIN' in row['db']:
                        index_lookup[row['filepath_deg']] = row['mos']
                        line_count += 1
                else:
                    if 'val' in row['db'] or 'VAL' in row['db']:
                        index_lookup[row['filepath_deg']] = row['mos']
                        line_count += 1
            else:
                 if Eval in row['db']:
                        index_lookup[row['filepath_deg']] = row['mos']
                        line_count += 1
                
    return index_lookup

def make_name_dict(label_csv, isTrain, Eval):
    name_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            if Eval == None:
                if isTrain:
                    if 'train' in row['db'] or 'TRAIN' in row['db']:
                        name_lookup[line_count] = row['filepath_deg']
                        line_count += 1
                else:
                    if 'val' in row['db'] or 'VAL' in row['db']:
                        name_lookup[line_count] = row['filepath_deg']
                        line_count += 1
            else:
                if Eval in row['db']:
                    name_lookup[line_count] = row['filepath_deg']
                    line_count += 1
    return name_lookup

def lookup_list(index_list, label_csv):
    label_list = []
    table = make_name_dict(label_csv)
    for item in index_list:
        label_list.append(table[item])
    return label_list

def preemphasis(signal,coeff=0.97):
    """perform preemphasis on the input signal.

    :param signal: The signal to filter.
    :param coeff: The preemphasis coefficient. 0 is none, default 0.97.
    :returns: the filtered signal.
    """
    return np.append(signal[0],signal[1:]-coeff*signal[:-1])

class NisqaDataset(Dataset):
    def __init__(self, datapath, audio_conf, label_csv=None, isTrain=True, Eval=None):
        """
        Dataset that manages audio recordings
        :param audio_conf: Dictionary containing the audio loading and preprocessing settings
        :param dataset_json_file
        """
        # self.datapath = dataset_json_file
        # with open(dataset_json_file, 'r') as fp:
        #     data_json = json.load(fp)

        # self.data = data_json['data']
        self.audio_conf = audio_conf
        self.datapath = datapath
        print('---------------the {:s} dataloader---------------'.format(self.audio_conf.get('mode')))
        self.melbins = self.audio_conf.get('num_mel_bins')
        self.freqm = self.audio_conf.get('freqm')
        self.timem = self.audio_conf.get('timem')
        print('now using following mask: {:d} freq, {:d} time'.format(self.audio_conf.get('freqm'), self.audio_conf.get('timem')))
        self.mixup = self.audio_conf.get('mixup')
        print('now using mix-up with rate {:f}'.format(self.mixup))
        self.dataset = self.audio_conf.get('dataset')
        print('now process ' + self.dataset)
        # dataset spectrogram mean and std, used to normalize the input
        self.norm_mean = self.audio_conf.get('mean')
        self.norm_std = self.audio_conf.get('std')
        # skip_norm is a flag that if you want to skip normalization to compute the normalization stats using src/get_norm_stats.py, if Ture, input normalization will be skipped for correctly calculating the stats.
        # set it as True ONLY when you are getting the normalization stats.
        self.skip_norm = self.audio_conf.get('skip_norm') if self.audio_conf.get('skip_norm') else False
        if self.skip_norm:
            print('now skip normalization (use it ONLY when you are computing the normalization stats).')
        else:
            print('use dataset mean {:.3f} and std {:.3f} to normalize the input.'.format(self.norm_mean, self.norm_std))
        # if add noise for data augmentation
        self.noise = self.audio_conf.get('noise')
        if self.noise == True:
            print('now use noise augmentation')

        self.index_dict = make_index_dict(label_csv, isTrain, Eval)
        self.name_dict = make_name_dict(label_csv, isTrain, Eval)
        self.label_num = len(self.index_dict)
        print('number of wavs is {:d}'.format(self.label_num))

    def _wav2fbank(self, filename, filename2=None):
        # mixup
        if filename2 == None:
            waveform, sr = torchaudio.load(filename)
            waveform = waveform - waveform.mean()
        # mixup
        else:
            waveform1, sr = torchaudio.load(filename)
            waveform2, _ = torchaudio.load(filename2)

            waveform1 = waveform1 - waveform1.mean()
            waveform2 = waveform2 - waveform2.mean()

            if waveform1.shape[1] != waveform2.shape[1]:
                if waveform1.shape[1] > waveform2.shape[1]:
                    # padding
                    temp_wav = torch.zeros(1, waveform1.shape[1])
                    temp_wav[0, 0:waveform2.shape[1]] = waveform2
                    waveform2 = temp_wav
                else:
                    # cutting
                    waveform2 = waveform2[0, 0:waveform1.shape[1]]

            # sample lambda from uniform distribution
            #mix_lambda = random.random()
            # sample lambda from beta distribtion
            mix_lambda = np.random.beta(10, 10)

            mix_waveform = mix_lambda * waveform1 + (1 - mix_lambda) * waveform2
            waveform = mix_waveform - mix_waveform.mean()


        
        fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                                                  window_type='hanning', num_mel_bins=self.melbins, dither=0.0, frame_shift=10)

        target_length = self.audio_conf.get('target_length')
        n_frames = fbank.shape[0]
        if self.audio_conf.get('padding_mode') == 'zero_padding':
            p = target_length - n_frames

            # cut and pad
            if p > 0:
                m = torch.nn.ZeroPad2d((0, 0, 0, p))
                fbank = m(fbank)
            elif p < 0:
                fbank = fbank[0:target_length, :]
        elif self.audio_conf.get('padding_mode') == 'repetitive':
            dup_times = target_length // n_frames
            remain = target_length - n_frames * dup_times
            to_dup = [fbank for t in range(dup_times)]
            to_dup.append(fbank[:remain, :])
            fbank = torch.Tensor(np.concatenate(to_dup, axis = 0))
        elif self.audio_conf.get('padding_mode') == 'noise':
            SNR = 30
            random_values = Tensor(np.random.rand(waveform.shape[1])).unsqueeze(dim = 0)
            # 计算语音信号功率Ps和噪声功率Pn1
            Ps = torch.sum(waveform ** 2) / waveform.shape[1]
            Pn1 = torch.sum(random_values ** 2) / random_values.shape[1]

            # 计算k值
            k = math.sqrt(Ps / (10 ** (SNR / 10) * Pn1))
            # 将噪声数据乘以k,
            random_values_we_need = random_values * k

            noise_fbank = torchaudio.compliance.kaldi.fbank(random_values_we_need, htk_compat=True, sample_frequency=sr, use_energy=False,
                                                  window_type='hanning', num_mel_bins=self.melbins, dither=0.0, frame_shift=10)

            dup_times = target_length // n_frames
            remain = target_length - n_frames * dup_times
            to_dup = [noise_fbank for t in range(dup_times)]
            to_dup.append(fbank[:remain, :])
            fbank = torch.Tensor(np.concatenate(to_dup, axis = 0))




        if filename2 == None:
            return fbank, 0
        else:
            return fbank, mix_lambda

    def __getitem__(self, index):
        """
        returns: image, audio, nframes
        where image is a FloatTensor of size (3, H, W)
        audio is a FloatTensor of size (N_freq, N_frames) for spectrogram, or (N_frames) for waveform
        nframes is an integer
        """
        # do mix-up for this sample (controlled by the given mixup rate)
        if random.random() < self.mixup:
            datum = self.data[index]
            # find another sample to mix, also do balance sampling
            # sample the other sample from the multinomial distribution, will make the performance worse
            # mix_sample_idx = np.random.choice(len(self.data), p=self.sample_weight_file)
            # sample the other sample from the uniform distribution
            mix_sample_idx = random.randint(0, len(self.data)-1)
            mix_datum = self.data[mix_sample_idx]
            # get the mixed fbank
            fbank, mix_lambda = self._wav2fbank(datum['wav'], mix_datum['wav'])
            # initialize the label
            label_indices = np.zeros(self.label_num)
            # add sample 1 labels
            for label_str in datum['labels'].split(','):
                label_indices[int(self.index_dict[label_str])] += mix_lambda
            # add sample 2 labels
            for label_str in mix_datum['labels'].split(','):
                label_indices[int(self.index_dict[label_str])] += 1.0-mix_lambda
            label_indices = torch.FloatTensor(label_indices)
        # if not do mixup
        else:
            # datum = self.data[index]
            # label_indices = np.zeros(self.label_num)
            # fbank, mix_lambda = self._wav2fbank(datum['wav'])
            # for label_str in datum['labels'].split(','):
            #     label_indices[int(self.index_dict[label_str])] = 1.0

            # label_indices = torch.FloatTensor(label_indices)
            fbank, mix_lambda = self._wav2fbank(os.path.join(self.datapath, self.name_dict.get(index)))
            # for label_str in datum['labels'].split(','):
            #     label_indices[int(self.index_dict[label_str])] = 1.0
            score = self.index_dict.get(self.name_dict.get(index))
            label_indices = float(score)
        # SpecAug, not do for eval set
        freqm = torchaudio.transforms.FrequencyMasking(self.freqm)
        timem = torchaudio.transforms.TimeMasking(self.timem)
        fbank = torch.transpose(fbank, 0, 1)
        # this is just to satisfy new torchaudio version, which only accept [1, freq, time]
        fbank = fbank.unsqueeze(0)
        if self.freqm != 0:
            fbank = freqm(fbank)
        if self.timem != 0:
            fbank = timem(fbank)
        # squeeze it back, it is just a trick to satisfy new torchaudio version
        fbank = fbank.squeeze(0)
        fbank = torch.transpose(fbank, 0, 1)

        # normalize the input for both training and test
        if not self.skip_norm:
            fbank = (fbank - self.norm_mean) / (self.norm_std * 2)
        # skip normalization the input if you are trying to get the normalization stats.
        else:
            pass

        if self.noise == True:
            fbank = fbank + torch.rand(fbank.shape[0], fbank.shape[1]) * np.random.rand() / 10
            fbank = torch.roll(fbank, np.random.randint(-10, 10), 0)

        mix_ratio = min(mix_lambda, 1-mix_lambda) / max(mix_lambda, 1-mix_lambda)

        # the output fbank shape is [time_frame_num, frequency_bins], e.g., [1024, 128]
        return fbank, label_indices

    def __len__(self):
        return self.label_num
