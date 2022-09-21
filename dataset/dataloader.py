import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from config.audio_twins_config import TwinsConfig
from utils import AudioTools

args = TwinsConfig()

class LibrispeechDataset(Dataset):
    def __init__(self):
        # data = pd.read_csv(args.wav_list_path)
        self.file_list = list(pd.read_csv(args.wav_list_path).iloc[:, 1])
        print('get {} file in dataloader'.format(len(self.file_list)))
        self.len = len(self.file_list)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        file_list = []
        feat_list = []
        file = self.file_list[idx]
        file_list.append(file)
        for file in file_list:
            sig, sr = AudioTools.load_wav(file, 16000)
            if len(sig) > args.max_len:
                sig = sig[:args.max_len]
            sig = np.array(sig)
            feat_list.append(sig)
        return {'fid': file_list, 'feat': feat_list}

def collate_fn(data):
    fids = [d['fid'] for d in data]
    feats = [d['feat'] for d in data]
    feats_batch = [torch.Tensor(np.array(feat)).squeeze() for feat in feats]
    feats_batch = torch.nn.utils.rnn.pad_sequence(feats_batch)
    return {'fid': fids,
            'feat': feats_batch.float()}