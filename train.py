from datasets import load_dataset
from s3prl.downstream.asr.dictionary import Dictionary
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import Wav2Vec2Model, BertModel, AdamW, Wav2Vec2Processor, BertTokenizer
from transformers import EarlyStoppingCallback

from dataset.s3prl_dataset import SequenceDataset
from models.model import JointModel, MaxPoolFusion
from utils.trainer import MyTrainer

import torch
import torch.nn as nn
import numpy as np


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.flatten()
    labels = labels.flatten()
    return {
        'pearson': np.corrcoef(predictions, labels)[0][1]
    }


def data_collate(batch):
    audio = [val['audio']['array'] for val in batch]
    text = [val['text'] for val in batch]
    processor = Wav2Vec2Processor.from_pretrained("./pretrain_models/wav2vec2-base-960h")
    tokenizer = BertTokenizer.from_pretrained('./pretrain_models/bert-base-cased')
    audio_feat = processor(audio, return_tensors="pt", padding="longest").input_values
    text_ids = []
    token_ids = []
    attn_mask = []
    for val in text:
        text_feat = tokenizer(val, return_tensors='pt')
        text_ids.append(text_feat['input_ids'].squeeze())
        token_ids.append(text_feat['token_type_ids'].squeeze())
        attn_mask.append(text_feat['attention_mask'].squeeze())

    text_ids_feat = pad_sequence(text_ids).T
    token_ids_feat = pad_sequence(token_ids).T
    attn_mask_feat = pad_sequence(attn_mask).T
    return {'audio': audio_feat, 'text_ids': text_ids_feat, 'token_ids': token_ids_feat, 'attn_mask': attn_mask_feat}


def compute_loss(input, target):
    loss_fn = nn.MSELoss()
    loss = loss_fn(input, target)
    return loss


if __name__ == "__main__":
    args = {
        'test-clean': ['test-clean'],
        'bucket_size': 10,
        'bucket_file': './dataset/data'
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    text_encoder = BertModel.from_pretrained('./pretrain_models/bert-base-cased')
    tokenizer = BertTokenizer.from_pretrained('./pretrain_models/bert-base-cased')
    audio_encoder = Wav2Vec2Model.from_pretrained("./pretrain_models/wav2vec2-base-960h")
    model = JointModel(audio_encoder, MaxPoolFusion()).to(device)

    # load dummy dataset and read soundfiles
    print('begin to load data')
    # ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
    ds = SequenceDataset(split='test-clean', dictionary=Dictionary(), text_encoder=text_encoder, tokenizer=tokenizer,
                              libri_root='F:\OneDrive\数据集\Librispeech\\test-clean\LibriSpeech',
                              **args)

    dataloader = torch.utils.data.DataLoader(ds, batch_size=1, collate_fn=ds.collate_fn)
    es = EarlyStoppingCallback(early_stopping_patience=5)
    optimizer = AdamW(model.parameters(), lr=5e-5)

    for epoch in range(5):
        step = 1
        for batch in tqdm(dataloader):
            audio_input = batch[0].to(device)
            text_feat = batch[4].to(device)
            loss, audio, text = model(audio_input, text_feat)
            print('step: {}, loss :{}'.format(step, loss))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            step += 1
