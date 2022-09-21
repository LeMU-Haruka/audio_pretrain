from transformers import BertTokenizer
from transformers import Wav2Vec2Processor
from torch.utils.data import Dataset


class LS_dataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        audio_wav = data['audio']['array']
        text = data['text']
        audio_input = self.processor(audio_wav, return_tensors="pt", padding="longest").input_values
        text_input = self.tokenizer(text, return_tensors='pt')
        return {'audio': audio_input,
                'text': text_input}