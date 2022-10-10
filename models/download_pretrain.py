from transformers import BertModel, BertTokenizer, Wav2Vec2Model

text_encoder = BertModel.from_pretrained('bert-base-cased')
text_encoder.save_pretrained('./pretrain_models/bert-base-cased')
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
tokenizer.save_pretrained('./pretrain_models/bert-base-cased')
audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
audio_encoder.save_pretrained('./pretrain_models/wav2vec2-base-960h')