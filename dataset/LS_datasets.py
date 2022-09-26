# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ dataset.py ]
#   Synopsis     [ the phone dataset ]
#   Author       [ S3PRL, Xuankai Chang ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
from torch import Tensor

"""*********************************************************************************************"""

###############
# IMPORTATION #
###############
import logging
import os
import random
# -------------#
import pandas as pd
from tqdm import tqdm
from pathlib import Path
# -------------#
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import Dataset
# -------------#
import torchaudio
# -------------#
import numpy as np

SAMPLE_RATE = 16000
HALF_BATCHSIZE_TIME = 2000


####################
# Sequence Dataset #
####################
class SequenceDataset(Dataset):

    def __init__(self, libri_root, bucket_file):
        super(SequenceDataset, self).__init__()

        self.libri_root = libri_root
        self.sample_rate = SAMPLE_RATE
        # Wavs
        table_list = []
        file_path = os.path.join(bucket_file)
        if os.path.exists(file_path):
            table_list.append(
                pd.read_csv(file_path)
            )

        table_list = pd.concat(table_list)
        table_list = table_list.sort_values(by=['length'], ascending=False)

        self.X = table_list['file_path'].tolist()
        self.X_lens = table_list['length'].tolist()

        # Transcripts
        Y = self._load_transcript(self.X)

        x_names = set([self._parse_x_name(x) for x in self.X])
        y_names = set(Y.keys())
        usage_list = list(x_names & y_names)

        Y = {key: Y[key] for key in usage_list}

        self.Y = Y

    def _parse_x_name(self, x):
        return x.split('/')[-1].split('.')[0]

    def _load_wav(self, wav_path):
        wav, sr = torchaudio.load(os.path.join(self.libri_root, wav_path))
        assert sr == self.sample_rate, f'Sample rate mismatch: real {sr}, config {self.sample_rate}'
        return wav.view(-1)

    def _load_transcript(self, x_list):
        """Load the transcripts for Librispeech"""

        def process_trans(transcript):
            # TODO: support character / bpe
            transcript = transcript.upper()
            return transcript

        trsp_sequences = {}
        split_spkr_chap_list = list(
            set(
                "/".join(x.split('/')[:-1]) for x in x_list
            )
        )

        for dir in split_spkr_chap_list:
            parts = dir.split('/')
            trans_path = f"{parts[-2]}-{parts[-1]}.trans.txt"
            path = os.path.join(self.libri_root, dir, trans_path)
            assert os.path.exists(path)

            with open(path, "r") as trans_f:
                for line in trans_f:
                    lst = line.strip().split()
                    trsp_sequences[lst[0]] = process_trans(" ".join(lst[1:]))

        return trsp_sequences

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        # Load acoustic feature and pad
        file = self.X[index]
        wav = self._load_wav(file).numpy()
        length = self.X_lens[index]
        text = self.Y[self._parse_x_name(file)]
        filename = Path(file).stem
        return {'wav': wav, 'text': text, 'length': length, 'filename': filename}

    def collate_fn(self, items):
        assert len(items) == 1
        return items[0][0], items[0][1], items[0][2]  # hack bucketing, return (wavs, labels, filenames)

def collate_fn(data):
    wavs = [d['wav'] for d in data]
    text = [d['text'] for d in data]
    length = [d['length'] for d in data]
    filename = [d['filename'] for d in data]
    wavs_batch = [torch.Tensor(np.array(feat)).squeeze() for feat in wavs]
    wavs_batch = torch.nn.utils.rnn.pad_sequence(wavs_batch)
    return {'wav': wavs_batch,
            'text': text,
            'length': length,
            'filename': filename}


# dataset = SequenceDataset('F:\OneDrive\数据集\Librispeech\\test-clean\LibriSpeech', './data/test-clean.csv')
#
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=5, collate_fn=collate_fn)
#
#
# for item in dataloader:
#     print('done')
#
# print('done')