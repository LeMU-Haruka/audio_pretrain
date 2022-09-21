import torch
import torch.nn as nn
import numpy as np
import tqdm
from transformers import Trainer


class MyTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        audio_input = inputs['audio']
        text_input = inputs['text']
        audio_feat, text_feat = model(audio_input, text_input)

        loss = self.loss(audio_feat, text_feat)
        return (loss, {'audio': audio_feat, 'text': text_feat}) if return_outputs else loss

    def loss(self, input, target):
        loss_fn = nn.MSELoss()
        loss = loss_fn(input, target)
        return loss
