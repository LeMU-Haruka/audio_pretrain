import argparse
import os.path
import shutil

import yaml

from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import Wav2Vec2Model, BertModel, Wav2Vec2Processor, BertTokenizer, BartConfig
from transformers import EarlyStoppingCallback
from torch.cuda.amp import GradScaler

from dataset.LS_datasets import SequenceDataset
from models.model import JointModel

import os
import torch
import numpy as np

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.flatten()
    labels = labels.flatten()
    return {
        'pearson': np.corrcoef(predictions, labels)[0][1]
    }


def bert_encode(encoder, text_input):
    with torch.no_grad():
        text_feat = [encoder(**val[0]).last_hidden_state for val in text_input]
    real = [val[3] for val in text_input]
    mask = [val[2] for val in text_input]
    return text_feat, mask, real


def load_config(file='./config/config.yaml'):
    config = BartConfig()
    with open(file, 'r') as f:
        yaml_conf = yaml.load(f, Loader=yaml.FullLoader)
    for key, value in yaml_conf.items():
        config.__setattr__(key, value)
    if not os.path.exists(config.output_path):
        os.makedirs(config.output_path)
    shutil.copy(file, config.output_path)
    print('already copy config to {}'.format(config.output_path))
    return config


def parse_args():
    """Parse args."""
    parser = argparse.ArgumentParser(description='Audio pretrain')
    parser.add_argument('--config_file', type=str, default="./config/config.yaml", help='Config file')
    args = parser.parse_known_args()[0]
    return args


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config_file)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config.device = device
    text_encoder = BertModel.from_pretrained('./pretrain_models/bert-base-cased')
    tokenizer = BertTokenizer.from_pretrained('./pretrain_models/bert-base-cased')
    config.vocab_size = tokenizer.vocab_size

    model = JointModel(config)
    # init prediction layer weight with bert embedding
    if config.is_init_pred_weight:
        model.update_pred_weight(text_encoder.embeddings.word_embeddings)

    # load dummy dataset and read soundfiles
    print('begin to load data')
    # ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
    ds = SequenceDataset(libri_root=config.librispeech_path, bucket_dir=config.bucket_dir,
                         bucket_file=config.bucket_file, tokenizer=tokenizer, text_encoder=text_encoder, config=config)

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
    for epoch in range(config.epoch):
        if epoch > 0:
            ds.modal_mask = True
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
        torch.save(model.encoder.state_dict(), os.path.join(config.output_path, 'fusion_{}.ckpt'.format(epoch)))
        # torch.save(model.fusion.state_dict(), 'trans_{}.pt'.format(epoch))
        print('epoch {} finished, save model: fusion_{}.ckpt'.format(epoch, epoch))
        torch.cuda.empty_cache()
    # f.close()
