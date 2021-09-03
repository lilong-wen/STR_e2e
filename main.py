import random
import torch.distributed as dist
import os
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
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

def get_gin_par(n):

    return gin.query_parameter(f'%{n}')


def gin_init(opt):

    gin.parse_config_file(opt.gin)

def main(args):

    model, criterion = build_model()
    model.to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr = get_gin_par('lr'),
                                  weight_decay = get_gin_par('weight_decay'))
    if args.dist == 'DDP':
        model = DDP(model, device_ids=[args.rank])

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   get_gin_par('lr_drop'))

    dataset_train, dataset_val = build_dataset()

    if args.dist == 'DDP':
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    '''
    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train,
        get_gin_par('batch_size'),
        drop_last=True)

    data_loader_train = DataLoader(
        dataset_train,
        batch_sampler=batch_sampler_train,
        collate_fn=utils.collate_fn,
        num_workers=get_gin_par('num_workers'))
    data_loader_val = DataLoader(
        dataset_val,
        get_gin_par('batch_size'),
        sampler=sampler_val,
        drop_last=False,
        num_workers=get_gin_par('num_workers'))
    '''
    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, get_gin_par('batch_size'), drop_last=True)

    data_loader_train = DataLoader(dataset_train,
                                   batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn,
                                   num_workers=get_gin_par('num_workers'))
    data_loader_val = DataLoader(dataset_val,
                                 get_gin_par('batch_size'),
                                 sampler=sampler_val,
                                 drop_last=False,
                                 collate_fn=utils.collate_fn,
                                 num_workers=get_gin_par('num_workers'))


    print(f"rank: {args.rank}")
    for item in data_loader_train:

        print(item[1])
        exit()
    exit()

    output_dir = get_gin_par('save_path')

    # start training
    print('star training')
    start_time = time.time()

    for epoch in range(0, 10):
        print(f"epoch: {epoch}")
        train_stats = train_one_epoch(
            model,
            criterion,
            data_loader_train,
            optimizer,
            args.device,
            epoch,
            0.1)
        lr_scheduler.step()
        print(f"epoch ended: {epoch}")
        # save weights
        checkpoint_paths = [output_dir / 'checkpoint.pth']
        if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 100 == 0:
            checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
        for checkpoint_path in checkpoint_paths:
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
            }, checkpoint_path)

        test_stats = evaluate(
            model,
            criterion,
            data_loader_val,
            base_ds,
            args.device,
            output_dir
        )
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

def rSeed(sd):
    random.seed(sd)
    np.random.seed(sd)
    torch.manual_seed(sd)
    torch.cuda.manual_seed(sd)

def launch_fn(rank, args):

    gin_init(args)

    mp.set_start_method('fork', force=True)

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(args.port)

    dist.init_process_group("nccl", rank=rank, world_size=args.num_gpu)

    #to ensure identical init parameters
    rSeed(args.manualSeed)

    torch.cuda.set_device(rank)
    args.world_size = args.num_gpu
    args.rank       = rank

    main(args)


if __name__ == '__main__':

    parser = argparse.ArgumentParser("config")
    parser.add_argument('--gin', default='config/ic15.gin', type=str)
    args = parser.parse_args()
    gin_init(args)
    args.port = get_gin_par("port")
    args.num_gpu = 2
    args.manualSeed = get_gin_par('manualSeed')
    args.dist = get_gin_par("dist")
    args.device = 'cuda' if torch.cuda.is_available else 'cpu'

    mp.spawn(launch_fn, args=(args,), nprocs=args.num_gpu)
