from typing import Tuple

import random
import torch
import torch.nn as nn

from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence
from transformers import Wav2Vec2Model, HubertModel, WavLMModel
from transformers.activations import ACT2FN

from models.specaug import SpecAug
from models.transformer import CrossTransformer


class MaxPool(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, feats):
        out = torch.max(feats, dim=1)[0]
        return out


class MaxPoolFusion(nn.Module):
    def __init__(self):
        super(MaxPoolFusion, self).__init__()
        self.audio_pool = MaxPool()
        self.text_pool = MaxPool()
        self.loss = nn.MSELoss()
        self.audio_bn = nn.BatchNorm1d(768)
        self.text_bn = nn.BatchNorm1d(768)

    def forward(self, audio, text):
        audio_pool = self.audio_pool(audio)
        text_pool = self.text_pool(text)
        audio_pool = self.audio_bn(audio_pool)
        text_pool = self.text_bn(text_pool)
        return self.loss(audio_pool, text_pool), audio_pool, text_pool


class JointModel(nn.Module):

    def __init__(self, config):
        super(JointModel, self).__init__()
        self.config = config
        self.encoder = FeatureFusionModel(config).to(config.device)
        self.prediction = PredictionModel(config).to(config.device)

    def forward(self, audio, text_feat, label, mask_index):
        x, audio_len = self.encoder(audio, text_feat)
        loss = self.prediction(x, audio_len, label, mask_index)
        return loss

    def update_pred_weight(self, embedding):
        weight = embedding.weight
        self.prediction.decoder.weight = weight


class FeatureFusionModel(nn.Module):

    def __init__(self, config):
        super(FeatureFusionModel, self).__init__()
        self.config = config
        if config.encoder == 'wav2vec2':
            self.audio_encoder = Wav2Vec2Model.from_pretrained(config.wav2vec_dir).to(config.device)
            self.audio_encoder.freeze_feature_extractor()
        if config.encoder == 'wavlm':
            self.audio_encoder = WavLMModel.from_pretrained(config.wav2vec_dir).to(config.device)
            self.audio_encoder.freeze_feature_extractor()
        if config.encoder == 'hubert':
            self.audio_encoder = HubertModel.from_pretrained(config.wav2vec_dir).to(config.device)
            self.audio_encoder.feature_extractor._freeze_parameters()

        if not config.is_train_wav2vec:
            self.audio_encoder.eval()
            self.freeze_wav2vec2()
        print('Will use {} as audio encoder'.format(config.encoder))
        # self.spec_aug = SpecAug()
        self.fusion = CrossTransformer(config).to(config.device)

    def forward(self, audio, text):
        fusion_feat = self.encode_features(audio, text)
        return fusion_feat

    def encode_features(self, audio, text):
        if self.config.is_train_wav2vec:
            audio_feat = [self.audio_encoder(val.to(self.audio_encoder.device)).last_hidden_state for val in audio]
        else:
            with torch.no_grad():
                audio_feat = [self.audio_encoder(val.to(self.audio_encoder.device)).last_hidden_state for val in audio]
        # if self.config.is_spec_aug:
        #     audio_feat = self.spec_aug(audio_feat)
        features = [torch.cat([a.squeeze(), t.squeeze().to(a.device)], 0) for a, t in
                    zip(audio_feat, text)]
        features = pad_sequence(features).transpose(0, 1)
        audio_len = [val.shape[1] for val in audio_feat]
        features = self.fusion(features)
        return features, audio_len

    def freeze_wav2vec2(self):
        for p in self.audio_encoder.parameters():
            p.requires_grad = False


class PredictionModel(nn.Module):

    def __init__(self, config):
        super(PredictionModel, self).__init__()
        self.config = config
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=True)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias
        self.gradient_checkpointing = False
        self.counter = 1

    def forward(self, x, audio_length, mask_index, x_real):
        x = self.transform(x)
        prediction_scores = self.decoder(x)
        loss_fct = CrossEntropyLoss(ignore_index=0)
        # TODO 只提取text部分的信息，注意非mask部分的label应该为-100，mask部分的label为真实label
        loss = 0
        for score, a_l, real in zip(prediction_scores, audio_length, x_real):
            if not real.any():
                print('no mask jumped {}'.format(real))
                continue
            mask_score = score[a_l:, :]
            if mask_score.shape[0] > real.shape[1]:
                mask_score = mask_score[:real.shape[1], :]
            if self.counter % 5000 == 0:
                self.counter = 0
                temp = nn.functional.softmax(mask_score.view(-1, self.config.vocab_size))
                index = torch.argmax(temp, 1)
                print('Compare decoder {}'.format(index))
                print('Real output     {}'.format(real))
            temp_loss = loss_fct(mask_score.view(-1, self.config.vocab_size), real.view(-1).to(mask_score.device))
            loss += temp_loss
            self.counter += 1
        return loss


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states
