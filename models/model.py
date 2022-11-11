from typing import Tuple

import random
import torch
import torch.nn as nn

from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence
from transformers import Wav2Vec2Model
from transformers.activations import ACT2FN

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
        self.replay_prediction = ReplayPredictionModel(config).to(config.device)

    def forward(self, audio, text_feat, label, mask_index, is_replay):
        if is_replay:
            feat, mask, origin = self.encoder(audio, None, is_replay)
            loss = self.replay_prediction(feat, mask, origin)
            return loss
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
        self.audio_encoder = Wav2Vec2Model.from_pretrained(config.wav2vec_dir).to(config.device)
        self.audio_encoder.freeze_feature_extractor()
        self.fusion = CrossTransformer(config).to(config.device)
        self.masked_spec_embed = nn.Parameter(torch.FloatTensor(768).uniform_())

    def forward(self, audio, text, is_replay=False):
        if is_replay:
            return self.replay_encode(audio)
        fusion_feat = self.encode_features(audio, text)
        return fusion_feat

    def replay_encode(self, audio):
        audio_feat = [self.audio_encoder(val.to(self.audio_encoder.device)).last_hidden_state for val in audio]
        features = []
        masks = []
        for val in audio_feat:
            masked, mask = self.mask(val)
            feat = self.fusion(masked)
            masks.append(mask)
            features.append(feat)
        return features, masks, audio_feat

    def encode_features(self, audio, text):
        if self.config.is_train_wav2vec:
            audio_feat = [self.audio_encoder(val.to(self.audio_encoder.device)).last_hidden_state for val in audio]
        else:
            with torch.no_grad():
                audio_feat = [self.audio_encoder(val.to(self.audio_encoder.device)).last_hidden_state for val in audio]

        features = [torch.cat([a.squeeze(), t.squeeze().to(a.device)], 0) for a, t in
                    zip(audio_feat, text)]
        features = pad_sequence(features).transpose(0, 1)
        audio_len = [val.shape[1] for val in audio_feat]
        features = self.fusion(features)
        return features, audio_len

    def mask(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = _compute_mask((x.size(0), x.size(1)), 0.2, 10, x.device, 2)
        x[mask] = self.masked_spec_embed.to(x.dtype)
        return x, mask


class ReplayPredictionModel(nn.Module):

    def __init__(self, config):
        super(ReplayPredictionModel, self).__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.bias = nn.Parameter(torch.zeros(config.hidden_size))
        self.decoder.bias = self.bias
        self.l2_loss = nn.MSELoss()

    def forward(self, x, masks, origin):
        loss = 0
        for i, m, o in zip(x, masks, origin):
            feat = self.transform(i)
            pred_feat = self.decoder(feat)
            l = self.l2_loss(pred_feat[m], o[m])
            loss += l
        return loss / len(x)


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


def _compute_mask(
        shape: Tuple[int, int],
        mask_prob: float,
        mask_length: int,
        device: torch.device,
        min_masks: int = 0,
) -> torch.Tensor:
    batch_size, sequence_length = shape

    if mask_length < 1:
        raise ValueError("`mask_length` has to be bigger than 0.")

    if mask_length > sequence_length:
        raise ValueError(
            f"`mask_length` has to be smaller than `sequence_length`, but got `mask_length`: {mask_length} and `sequence_length`: {sequence_length}`"
        )

    # compute number of masked spans in batch
    num_masked_spans = int(mask_prob * sequence_length / mask_length + random.random())
    num_masked_spans = max(num_masked_spans, min_masks)

    # make sure num masked indices <= sequence_length
    if num_masked_spans * mask_length > sequence_length:
        num_masked_spans = sequence_length // mask_length

    # SpecAugment mask to fill
    mask = torch.zeros((batch_size, sequence_length), device=device, dtype=torch.bool)

    # uniform distribution to sample from, make sure that offset samples are < sequence_length
    uniform_dist = torch.ones(
        (batch_size, sequence_length - (mask_length - 1)), device=device
    )

    # get random indices to mask
    mask_indices = torch.multinomial(uniform_dist, num_masked_spans)

    # expand masked indices to masked spans
    mask_indices = (
        mask_indices.unsqueeze(dim=-1)
            .expand((batch_size, num_masked_spans, mask_length))
            .reshape(batch_size, num_masked_spans * mask_length)
    )
    offsets = (
        torch.arange(mask_length, device=device)[None, None, :]
            .expand((batch_size, num_masked_spans, mask_length))
            .reshape(batch_size, num_masked_spans * mask_length)
    )
    mask_idxs = mask_indices + offsets

    # scatter indices to mask
    mask = mask.scatter(1, mask_idxs, True)

    return mask

#
# from transformers import BertModel, BertTokenizer
# bert = BertModel.from_pretrained('bert-base-cased')
# tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
# text = 'i am hello world'
# toked = tokenizer(text, return_tensors='pt')
# output = bert(**toked).last_hidden_state
# model = JointModel(None, None)
# test = model.mask_out(output, toked)
# # print('done')
#
# from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2Model
# from datasets import load_dataset
# import torch
#
# # load model and tokenizer
# processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
# # model1 = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
# # model2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
#
#
# ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
#
# # tokenize
# # input_values = processor(ds[0]["audio"]["array"], return_tensors="pt", padding="longest").input_values  # Batch size 1
#
# # retrieve logits
# # logits1 = model1.base_model(input_values)
# # logits2 = model2(input_values)
# print('done')
# tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
# test_data = ds[0]
# audio = test_data["audio"]['array']
# audio_tokened = processor(audio, return_tensors="pt", padding="longest").input_values
# text = test_data['text']
# text_tokened = tokenizer(text, return_tensors='pt')
# text_encoder = BertModel.from_pretrained('bert-base-cased')
# audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
# input_ids = text_tokened['input_ids']
# token_type_ids = text_tokened['token_type_ids']
# attention_mask = text_tokened['attention_mask']
# model = JointModel(audio_encoder, text_encoder)
# audio_feat, text_feat = model(audio_tokened, input_ids, token_type_ids, attention_mask)
# # loss = compute_loss(audio_feat, text_feat)
# print('done')
