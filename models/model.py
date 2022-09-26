import librosa
import torch
import torch.nn as nn
import numpy as np
import math


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

    def forward(self, audio, text):
        audio_pool = self.audio_pool(audio)
        text_pool = self.text_pool(text)
        return self.loss(audio_pool, text_pool), audio_pool, text_pool


class JointModel(nn.Module):

    def __init__(self, audio_encoder, fusion):
        super(JointModel, self).__init__()
        self.audio_encoder = audio_encoder
        self.fusion = fusion

    def forward(self, audio, text_feat):
        audio_feat = self.audio_encoder(audio).last_hidden_state
        audio_pool = self.audio_pool(audio_feat)
        loss, audio_pool, text_pool = self.fusion(audio_pool, text_feat)
        return loss, audio_pool, text_pool

    def mask_out(self, x, lengths):
        """
        Decide of random words to mask out, and what target they get assigned.
        """
        params = self.params
        slen, bs = x.size()

        # define target words to predict
        if params.sample_alpha == 0:
            pred_mask = np.random.rand(slen, bs) <= params.word_pred
            pred_mask = torch.from_numpy(pred_mask.astype(np.uint8))
        else:
            x_prob = params.mask_scores[x.flatten()]
            n_tgt = math.ceil(params.word_pred * slen * bs)
            tgt_ids = np.random.choice(len(x_prob), n_tgt, replace=False, p=x_prob / x_prob.sum())
            pred_mask = torch.zeros(slen * bs, dtype=torch.uint8)
            pred_mask[tgt_ids] = 1
            pred_mask = pred_mask.view(slen, bs)

        # do not predict padding
        pred_mask[x == params.pad_index] = 0
        pred_mask[0] = 0  # TODO: remove

        # mask a number of words == 0 [8] (faster with fp16)
        if params.fp16:
            pred_mask = pred_mask.view(-1)
            n1 = pred_mask.sum().item()
            n2 = max(n1 % 8, 8 * (n1 // 8))
            if n2 != n1:
                pred_mask[torch.nonzero(pred_mask).view(-1)[:n1 - n2]] = 0
            pred_mask = pred_mask.view(slen, bs)
            assert pred_mask.sum().item() % 8 == 0

        # generate possible targets / update x input
        _x_real = x[pred_mask]
        _x_rand = _x_real.clone().random_(params.n_words)
        _x_mask = _x_real.clone().fill_(params.mask_index)
        probs = torch.multinomial(params.pred_probs, len(_x_real), replacement=True)
        _x = _x_mask * (probs == 0).long() + _x_real * (probs == 1).long() + _x_rand * (probs == 2).long()
        x = x.masked_scatter(pred_mask, _x)

        assert 0 <= x.min() <= x.max() < params.n_words
        assert x.size() == (slen, bs)
        assert pred_mask.size() == (slen, bs)

        return x, _x_real, pred_mask
#
# from transformers import BertModel, BertTokenizer
# # bert = BertModel.from_pretrained('bert-base-cased')
# # tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
# # text = 'i am hello world'
# # toked = tokenizer(text, return_tensors='pt')
# # output = bert(**toked)
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
