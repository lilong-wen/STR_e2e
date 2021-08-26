import argparse
import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
from engine import evaluate, train_one_epoch
from network import build_model
from dataset import build_dataset
import gin

def get_gin_par(n):

    return gin.query_parameter(f'%{n}')

def get_args_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--gin', default='config/ic15.gin', type=str)

    return parser

def gin_init(opt):

    gin.parse_config_file(opt.gin)

def main(args):

    model, criterion = build_model()
    model.to(args.device)

    param_dicts = [
        {"param": [p for n, p in model.named_parameters()]}
    ]

    optimizer = torch.optim.AdamW(param_dicts,
                                  lr = get_gin_par('lr'),
                                  weight_decay = get_gin_par('weight_decay'))
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   get_gin_par('lr_drop'))

    dataset_train, dataset_val = build_dataset()
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train,
        get_gin_par('batch_size'),
        drop_last=True)

    data_loader_train = DataLoader(
        dataset_train,
        batch_sampler=batch_sampler_train,
        num_workers=get_gin_par('num_workers'))
    data_loader_val = DataLoader(
        dataset_val,
        get_gin_par('batch_size'),
        sampler=sampler_val,
        drop_last=False,
        num_workers=args.num_workers)

    output_dir = get_gin_par('save_path')

    # start training
    print('star training')
    start_time = time.time()

    for epoch in range(0, 10):

        train_stats = train_one_epoch(
            model,
            criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            0.1)
        lr_scheduler.step()

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
            device,
            output_dir
        )
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))




if __name__ == '__main__':

    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    args = parser.parse_args()
    gin_init(args)

    device = 'cuda' if torch.cuda.is_available else 'cpu'

    args.device = device
