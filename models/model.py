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


class FeatureFusionModel(nn.Module):

    def __init__(self, config):
        super(FeatureFusionModel, self).__init__()
        self.config = config
        self.audio_encoder = Wav2Vec2Model.from_pretrained(config.wav2vec_dir).to(config.device)
        self.audio_encoder.freeze_feature_extractor()
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

        features = [torch.cat([a.squeeze(), t.squeeze().to(a.device)], 0) for a, t in
                    zip(audio_feat, text)]
        features = pad_sequence(features).transpose(0, 1)
        audio_len = [val.shape[1] for val in audio_feat]
        features = self.fusion(features)
        return features, audio_len


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
                continue
            mask_score = score[a_l:, :]
            if mask_score.shape[0] > real.shape[1]:
                mask_score = mask_score[:real.shape[1], :]
            if self.counter % 3000 == 0:
                temp = nn.functional.softmax(mask_score.view(-1, self.config.vocab_size))
                index = torch.argmax(temp, 1)
                print('Compare decoder {} and real {}'.format(index, real))
            temp_loss = loss_fct(mask_score.view(-1, self.config.vocab_size), real.view(-1).to(mask_score.device))
            loss += temp_loss
        return loss
    #
    # def mask_out(self, x):
    #     """
    #     Decide of random words to mask out, and what target they get assigned.
    #     """
    #     x = x.squeeze()
    #     word_pred = 0.15
    #     fp16 = False
    #     params = {}
    #     sample_alpha = 0
    #     slen, bs = x.size()
    #
    #     # define target words to predict
    #     if sample_alpha == 0:
    #         pred_mask = np.random.rand(slen, bs) <= word_pred
    #         pred_mask = torch.from_numpy(pred_mask.astype(np.uint8))
    #     else:
    #         x_prob = params.mask_scores[x.flatten()]
    #         n_tgt = math.ceil(params.word_pred * slen * bs)
    #         tgt_ids = np.random.choice(len(x_prob), n_tgt, replace=False, p=x_prob / x_prob.sum())
    #         pred_mask = torch.zeros(slen * bs, dtype=torch.uint8)
    #         pred_mask[tgt_ids] = 1
    #         pred_mask = pred_mask.view(slen, bs)
    #
    #     # do not predict padding
    #     # 将batch padding的部分去掉不考虑
    #     # pred_mask[x == params.pad_index] = 0
    #     # pred_mask[0] = 0  # TODO: remove
    #
    #     # mask a number of words == 0 [8] (faster with fp16)
    #     if fp16:
    #         pred_mask = pred_mask.view(-1)
    #         n1 = pred_mask.sum().item()
    #         n2 = max(n1 % 8, 8 * (n1 // 8))
    #         if n2 != n1:
    #             pred_mask[torch.nonzero(pred_mask).view(-1)[:n1 - n2]] = 0
    #         pred_mask = pred_mask.view(slen, bs)
    #         assert pred_mask.sum().item() % 8 == 0
    #
    #     # generate possible targets / update x input
    #     _x_real = x[pred_mask]
    #     _x_rand = _x_real.clone().random_(self.config.vocab_size)
    #     _x_mask = _x_real.clone().fill_(params.mask_index)
    #     probs = torch.multinomial(params.pred_probs, len(_x_real), replacement=True)
    #     _x = _x_mask * (probs == 0).long() + _x_real * (probs == 1).long() + _x_rand * (probs == 2).long()
    #     x = x.masked_scatter(pred_mask, _x)
    #
    #     assert 0 <= x.min() <= x.max() < params.n_words
    #     assert x.size() == (slen, bs)
    #     assert pred_mask.size() == (slen, bs)
    #
    #     return x, _x_real, pred_mask
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
