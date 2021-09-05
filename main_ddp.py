#  python -m torch.distributed.launch --use_env main_ddp.py

import random
import torch.distributed as dist
import os
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from pathlib import Path
import argparse
import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
from engine import evaluate, train_one_epoch
from network import build_model
from datasets import build_dataset
import gin
import utils.misc as utils
import time

def gin_init(args):

    gin.parse_config_file(args.gin)

def get_gin_par(n):

    return gin.query_parameter(f'%{n}')

def random_Seed(sd):
    random.seed(sd)
    np.random.seed(sd)
    torch.manual_seed(sd)
    torch.cuda.manual_seed(sd)


def main(args):

    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    random_Seed(args.seed)

    model, criterion = build_model()
    model.to(device)

    model_without_ddp = model

    if args.distributed:

        model = DDP(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'number of parameters: {n_parameters}')

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]

    optimizer = torch.optim.AdamW(param_dicts,
                                  lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    dataset_train, dataset_val = build_dataset()

    if args.distributed:
        args.num_workers = 0
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train,
                                   batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn,
                                   num_workers=args.num_workers)


    data_loader_val = DataLoader(dataset_val,
                                 args.batch_size,
                                 sampler=sampler_val,
                                 drop_last=False,
                                 collate_fn=utils.collate_fn,
                                 num_workers=args.num_workers)


    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1


    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model,
            criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            args.clip_max_norm)
        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 100 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        exit()
        test_stats = evaluate(
            model, criterion, data_loader_val, device
        )

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":

    parser = argparse.ArgumentParser('config for gin')
    parser.add_argument('--gin', default='config/ic15.gin', type=str)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    opts = parser.parse_args()
    gin_init(opts)
    opts.dist_url = "env://"
    opts.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    opts.seed = get_gin_par('seed')
    opts.lr_backbone = get_gin_par('lr_backbone')
    opts.lr = get_gin_par('lr')
    opts.weight_decay = get_gin_par('weight_decay')
    opts.batch_size = get_gin_par('batch_size')
    opts.num_workers = get_gin_par('num_workers')
    opts.lr_drop = get_gin_par('lr_drop')
    opts.output_dir = get_gin_par('output_dir')
    opts.clip_max_norm = get_gin_par('clip_max_norm')
    opts.start_epoch = get_gin_par("start_epoch")
    opts.epochs = get_gin_par('epochs')
    main(opts)
