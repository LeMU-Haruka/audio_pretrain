from torch import Tensor, autograd
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from transformers import Wav2Vec2Model, BertModel, Wav2Vec2Processor, BertTokenizer, BartConfig
from transformers import EarlyStoppingCallback
from torch.cuda.amp import autocast as autocast, GradScaler

from dataset.LS_datasets import SequenceDataset
from models.model import JointModel

import torch
import torch.nn as nn
import numpy as np

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


config = BartConfig()
config.num_hidden_layers = 6
config.hidden_size = 768
config.encoder_ffn_dim = 2048
config.hidden_act = 'relu'
config.pad_index = 103
config.word_pred = 0.15
config.is_train_wav2vec=False
config.batch_size = 2
config.real_batch_size = 16
config.output_path='./output/'
config.wav2vec_dir='./pretrain_models/wav2vec2-base-960h'
# For yunnao
config.librispeech_path='/userhome/dataset/librispeech/LibriSpeech'
# for PC
config.librispeech_path='F:\OneDrive\数据集\Librispeech\\test-clean\LibriSpeech'


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.flatten()
    labels = labels.flatten()
    return {
        'pearson': np.corrcoef(predictions, labels)[0][1]
    }

def compute_loss(input, target):
    loss_fn = nn.MSELoss()
    loss = loss_fn(input, target)
    return loss


def bert_encode(encoder, text_input):
    with torch.no_grad():
        text_feat = [encoder(**val[0]).last_hidden_state for val in text_input]
    real_label = [val[3] for val in text_input]
    mask = [val[2] for val in text_input]
    return text_feat, mask, real_label

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config.device = device
    text_encoder = BertModel.from_pretrained('./pretrain_models/bert-base-cased')
    tokenizer = BertTokenizer.from_pretrained('./pretrain_models/bert-base-cased')
    config.vocab_size = tokenizer.vocab_size
    # tokenizer.save_pretrained('./pretrain_models/bert-large-uncased')
    # audio_encoder = Wav2Vec2Model.from_pretrained("./pretrain_models/wav2vec2-base-960h").to(device)
    model = JointModel(config)
    # init prediction weight with bert embedding
    # model.update_pred_weight(text_encoder.embeddings.word_embeddings)
    # load dummy dataset and read soundfiles
    print('begin to load data')
    # ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
    ds = SequenceDataset(libri_root=config.librispeech_path, bucket_dir='./dataset/data',
                         bucket_file=['test-clean'], tokenizer=tokenizer, text_encoder=text_encoder, config=config)

    dataloader = torch.utils.data.DataLoader(ds, batch_size=config.batch_size, num_workers=2, collate_fn=ds.collate_fn)
    es = EarlyStoppingCallback(early_stopping_patience=5)
    optimizer = AdamW(model.parameters(), lr=1e-6)

    step_size = len(dataloader)
    # file = './log.txt'
    # f = open(file, 'w')
    # f.write('begin to train model\n')
    scaler = GradScaler()

    acc_step = config.real_batch_size / config.batch_size

    print('begin to train model, total step is {}'.format(step_size))
    print('is_train_wav2vec is {}'.format(config.is_train_wav2vec))
    for epoch in range(10):
        if epoch > 0:
            ds.modal_mask=True
        print('new epoch run, modal mask is {}'.format(ds.modal_mask))
        step = 1
        print_loss = 0
        for batch in dataloader:
            audio_input = batch['wav']
            text_feature, mask, real_label = bert_encode(text_encoder, batch['text_feat'])

            loss = model(audio_input, text_feature, mask, real_label)

            loss = loss / acc_step
            print_loss += loss
            loss.backward()

            # 梯度累加
            if step % acc_step == 0:
                print('step: [{} / {}], loss :{}'.format(step, step_size, print_loss))
                print_loss = 0
                optimizer.step()
                optimizer.zero_grad()
            step += 1

        # torch.save(model.state_dict(), 'model_{}.pt'.format(epoch))
        torch.save(model.encoder.state_dict(), 'fusion_{}.ckpt'.format(epoch))
        # torch.save(model.fusion.state_dict(), 'trans_{}.pt'.format(epoch))
        print('epoch {} finished, save model: fusion_{}.ckpt'.format(epoch, epoch))
        torch.cuda.empty_cache()
    # f.close()
