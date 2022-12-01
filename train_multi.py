import argparse
import json
import math
import os
import random
import shutil
import signal
import subprocess
import sys
import time
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import argparse

import yaml
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import Wav2Vec2Model, BertModel, Wav2Vec2Processor, BertTokenizer, BartConfig
from transformers import EarlyStoppingCallback
from torch.cuda.amp import GradScaler
from contextlib import nullcontext

from dataset.LS_datasets import SequenceDataset
from models.model import JointModel

import os
import torch
import numpy as np

def init_seeds(seed=42, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True

def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.getenv("SLURM_JOB_ID")}')
    exit()

def handle_sigterm(signum, frame):
    pass

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

def bert_encode(encoder, text_input):
    with torch.no_grad():
        text_feat = [encoder(**val[0]).last_hidden_state for val in text_input]
    real = [val[3] for val in text_input]
    mask = [val[2] for val in text_input]
    return text_feat, mask, real


def main():
    args = load_config()
    args.ngpus_per_node = torch.cuda.device_count()
    if 'SLURM_JOB_ID' in os.environ:
        # single-node and multi-node distributed training on SLURM cluster
        # requeue job on SLURM preemption
        signal.signal(signal.SIGUSR1, handle_sigusr1)
        signal.signal(signal.SIGTERM, handle_sigterm)
        # find a common host name on all nodes
        # assume scontrol returns hosts in the same order on all nodes
        cmd = 'scontrol show hostnames ' + os.getenv('SLURM_JOB_NODELIST')
        stdout = subprocess.check_output(cmd.split())
        host_name = stdout.decode().splitlines()[0]
        args.rank = int(os.getenv('SLURM_NODEID')) * args.ngpus_per_node
        args.world_size = int(os.getenv('SLURM_NNODES')) * args.ngpus_per_node
        args.dist_url = f'tcp://{host_name}:58472'
    else:
        # single-node distributed training
        args.rank = 0
        args.dist_url = 'tcp://localhost:58472'
        args.world_size = args.ngpus_per_node
    torch.multiprocessing.spawn(main_worker, (args,), args.ngpus_per_node)

def main_worker(gpu, args):
    config = args
    args.rank += gpu
    torch.distributed.init_process_group(
        backend='nccl', init_method=args.dist_url,
        world_size=args.world_size, rank=args.rank)

    print('init done')
    if args.rank == 0:
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        stats_file = open(args.checkpoint_dir / 'stats.txt', 'a', buffering=1)
        print(' '.join(sys.argv))
        print(' '.join(sys.argv), file=stats_file)

    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True

    text_encoder = BertModel.from_pretrained('./pretrain_models/bert-base-cased')
    tokenizer = BertTokenizer.from_pretrained('./pretrain_models/bert-base-cased')
    args.vocab_size = tokenizer.vocab_size

    model = JointModel(args).cuda(gpu)

    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # init prediction layer weight with bert embedding
    if args.is_init_pred_weight:
        model.update_pred_weight(text_encoder.embeddings.word_embeddings)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu], find_unused_parameters=True)

    print('begin to load data')
    ds = SequenceDataset(libri_root=config.librispeech_path, bucket_dir=config.bucket_dir,
                         bucket_file=config.bucket_file, tokenizer=tokenizer, text_encoder=text_encoder, config=config)

    sampler = torch.utils.data.distributed.DistributedSampler(ds)
    assert args.batch_size % args.world_size == 0
    per_device_batch_size = args.batch_size // args.world_size
    loader = torch.utils.data.DataLoader(ds, batch_size=config.batch_size, num_workers=2,  pin_memory=True, sampler=sampler, collate_fn=ds.collate_fn)
    step_size = len(loader)

    # 优化后版本 要注意lr和数据的关系，100h1e-6就可以
    #optimizer = AdamW(model.parameters(), lr=1e-6)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-6)

    # automatically resume from checkpoint if it exists
    if (args.checkpoint_dir / 'checkpoint.pth').is_file():
        ckpt = torch.load(args.checkpoint_dir / 'checkpoint.pth',
                          map_location='cpu')
        start_epoch = ckpt['epoch']
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
    else:
        start_epoch = 0
    acc_step = config.real_batch_size / config.batch_size
    print('begin to train model, total step is {}'.format(step_size))
    print('is_train_wav2vec is {}'.format(config.is_train_wav2vec))

    start_time = time.time()
    # 自动混合精度
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(start_epoch, args.epoch):
        sampler.set_epoch(epoch)
        if epoch > 0:
            ds.modal_mask = True
        print_loss = 0
        for step, data in enumerate(loader, start=epoch * len(loader)):
            optimizer.zero_grad()
            my_context = model.no_sync if args.world_size != -1 and step % acc_step != 0 else nullcontext
            with my_context():
                is_replay = False
                audio_input = data['wav']
                text_feature, mask, real_label = bert_encode(text_encoder, data['text_feat'])

                if config.is_replay:
                    if random.random() < config.replay_prob:
                        is_replay = True

                with torch.cuda.amp.autocast():
                    loss = model(audio_input, text_feature, mask, real_label, is_replay)
                if isinstance(loss, torch.Tensor):
                    loss = loss / acc_step
                    print_loss += loss

                    scaler.scale(loss).backward()

            if step % acc_step == 0:
                scaler.step(optimizer)
                scaler.update()
            if step % args.print_freq == 0:
                if args.rank == 0:
                    stats = dict(epoch=epoch, step=step,
                                 lr_weights=optimizer.param_groups[0]['lr'],
                                 lr_biases=optimizer.param_groups[1]['lr'],
                                 loss=loss.item(),
                                 time=int(time.time() - start_time))
                    print(json.dumps(stats))
                    if args.rank == 0:
                        # save checkpoint
                        state = dict(epoch=epoch + 1, model=model.state_dict(),
                                     optimizer=optimizer.state_dict())
                        torch.save(state, args.checkpoint_dir / 'checkpoint.pt')
                    print(json.dumps(stats), file=stats_file)

        if args.rank == 0:
            # save final model
            torch.save(model.module.encoder.state_dict(),
                       os.path.join(config.output_path, 'fusion_{}.ckpt'.format(epoch)))


if __name__ == '__main__':
    init_seeds()
    main()