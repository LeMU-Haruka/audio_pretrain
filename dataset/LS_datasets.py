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

    def __init__(self, libri_root, bucket_dir, bucket_file, tokenizer, text_encoder, config):
        super(SequenceDataset, self).__init__()

        self.config = config
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.libri_root = libri_root
        self.sample_rate = SAMPLE_RATE
        # Wavs
        table_list = []
        for file in bucket_file:
            file_path = os.path.join(bucket_dir, (file + '.csv'))
            if os.path.exists(file_path):
                table_list.append(
                    pd.read_csv(file_path)
                )

        table_list = pd.concat(table_list)
        table_list = table_list.sort_values(by=['length'], ascending=False)

        self.X = table_list['file_path'].tolist()
        self.X_lens = table_list['length'].tolist()

        # Transcripts
        Y = self.load_transcript(self.X)

        x_names = set([self._parse_x_name(x) for x in self.X])
        y_names = set(Y.keys())
        usage_list = list(x_names & y_names)

        Y = {key: Y[key] for key in usage_list}

        self.Y = Y
        self.modal_mask = False
        # self.Y_t = {key : self.mask_out(self.Y[key], 0) for key in usage_list}

    def _parse_x_name(self, x):
        return x.split('/')[-1].split('.')[0]

    def _load_wav(self, wav_path):
        wav, sr = torchaudio.load(os.path.join(self.libri_root, wav_path))
        assert sr == self.sample_rate, f'Sample rate mismatch: real {sr}, config {self.sample_rate}'
        return wav.view(-1)

    def load_transcript(self, x_list):
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

    def encode_text(self, text):
        text_token = self.tokenizer(text, return_tensors='pt')
        # with torch.no_grad():
        #     text_feat = self.text_encoder(**text_token).last_hidden_state
        #     text_feat = text_feat.squeeze()
        text_labels, mask_label, mask, label, real = self.mask_out(text_token['input_ids'])
        text_token['input_ids'] = text_labels
        return text_token, mask_label, mask, label, real

    def __getitem__(self, index):
        # Load acoustic feature and pad
        file = self.X[index]
        wav = self._load_wav(file).unsqueeze(1).transpose(0, 1)
        audio_len = self.X_lens[index]
        text = self.Y[self._parse_x_name(file)]
        text_feat = self.encode_text(text)
        text_len = len(text.split(' '))
        filename = Path(file).stem
        return {'wav': wav, 'text_feat': text_feat, 'text': text, 'audio_len': audio_len, 'text_len': text_len,'filename': filename}

    def collate_fn(self, data):
        data = list(filter(None, data))
        wav_batch = [item['wav'] for item in data]
        # wav_batch = pad_sequence(wav_batch).transpose(0, 1)
        text_batch = [item['text_feat'] for item in data]
        # text_batch = pad_sequence(text_batch).transpose(0, 1)
        text = [item['text'] for item in data]
        audio_len = [item['audio_len'] for item in data]
        text_len = [item['text_len'] for item in data]
        filename = [item['filename'] for item in data]
        return {'wav': wav_batch, 'text_feat': text_batch,
                'text': text, 'audio_len': audio_len, 'text_len': text_len,'filename': filename}

    def mask_out(self, x):
        """
        Decide of random words to mask out, and what target they get assigned.
        input is [bs, seq_len]
        """
        origin = x.clone()
        fp16 = False
        mask_index = 103
        bs, slen = x.size()
        pred_mask = self.lm_mask(bs, slen)
        # if self.modal_mask:
        #     if random.random() < 0.5:
        #         pred_mask = self.modality_mask(bs, slen)

        word_mask, word_keep, word_rand = 0.8,0.1,0.1
        pred_probs = torch.FloatTensor([word_mask, word_keep, word_rand])

        # mask a number of words == 0 [8] (faster with fp16)
        if fp16:
            pred_mask = pred_mask.view(-1)
            n1 = pred_mask.sum().item()
            n2 = max(n1 % 8, 8 * (n1 // 8))
            if n2 != n1:
                pred_mask[torch.nonzero(pred_mask).view(-1)[:n1 - n2]] = 0
            pred_mask = pred_mask.view(slen, bs)
            assert pred_mask.sum().item() % 8 == 0

        # generate possible targets / update x input
        _x_real = x[pred_mask]
        _x_rand = _x_real.clone().random_(self.config.vocab_size)
        _x_mask = _x_real.clone().fill_(mask_index)

        if len(_x_real) == 0:
            return x, _x_real, pred_mask, pred_mask, origin
        probs = torch.multinomial(pred_probs, len(_x_real), replacement=True)
        _x = _x_mask * (probs == 0).long() + _x_real * (probs == 1).long() + _x_rand * (probs == 2).long()
        x = x.masked_scatter(pred_mask, _x)
        label = torch.tensor(pred_mask.clone(), dtype=torch.int64)
        label = label.masked_scatter(pred_mask, _x_real)
        # assert 0 <= x.min() <= x.max() < params.n_words
        # assert x.size() == (slen, bs)
        # assert pred_mask.size() == (slen, bs)

        return x, _x_real, pred_mask, label, origin

    def modality_mask(self, bs, slen):
        pred_mask = torch.ones(bs, slen, dtype=torch.uint8)
        return pred_mask

    def lm_mask(self, bs, slen):
        word_pred = self.config.word_pred
        pred_mask = np.random.rand(bs, slen) <= word_pred
        pred_mask = torch.from_numpy(pred_mask.astype(np.uint8))
        return pred_mask