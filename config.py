from transformers import BartConfig

class Config(BartConfig):

    def __init__(self):
        super(Config, self).__init__()
        self.num_hidden_layers = 3
        self.hidden_size = 768
        self.encoder_ffn_dim = 2048
        self.hidden_act = 'relu'
        self.pad_index = 103
        self.word_pred = 0.15
        self.is_train_wav2vec = False
        self.batch_size = 4
        self.real_batch_size = 16
