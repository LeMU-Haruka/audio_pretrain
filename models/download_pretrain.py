from transformers import BertModel, BertTokenizer, Wav2Vec2Model, HubertModel

# text_encoder = BertModel.from_pretrained('bert-base-cased')
# text_encoder.save_pretrained('./pretrain_models/bert-base-cased')
# tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
# tokenizer.save_pretrained('./pretrain_models/bert-base-cased')
# audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
# audio_encoder.save_pretrained('./pretrain_models/wav2vec2-base-960h')
#
# hubert = HubertModel.from_pretrained('facebook/hubert-base-ls960')
# hubert.save_pretrained('./pretrain_models/hubert-base-ls960')

hubert = HubertModel.from_pretrained('microsoft/wavlm-base')
hubert.save_pretrained('./pretrain_models/wavlm-base')