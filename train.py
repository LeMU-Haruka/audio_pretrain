from datasets import load_dataset
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import Wav2Vec2Model, BertModel, AdamW, Wav2Vec2Processor, BertTokenizer
from transformers import EarlyStoppingCallback

from dataset.datasets import LS_dataset
from models.model import JointModel
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
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    text_encoder = BertModel.from_pretrained('bert-base-cased')
    audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
    model = JointModel(audio_encoder, text_encoder).to(device)


    # load dummy dataset and read soundfiles
    ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
    dataloader = torch.utils.data.DataLoader(ds, batch_size=5, collate_fn=data_collate)

    es = EarlyStoppingCallback(early_stopping_patience=5)
    optimizer = AdamW(model.parameters(), lr=5e-5)

    for epoch in range(5):
        for batch in tqdm(dataloader):
            audio_input = batch['audio'].to(device)
            text_ids = batch['text_ids'].to(device)
            token_ids = batch['token_ids'].to(device)
            attn_mask = batch['attn_mask'].to(device)
            audio, text = model(audio_input, text_ids, token_ids, attn_mask)
            loss = compute_loss(audio, text)
            print('loss :{}'.format(loss))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()