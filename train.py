#!/usr/bin/env python

import os
import numpy as np
import pandas as pd
import wandb
import time
import argparse

import torch
import torch.nn as nn
import torch.multiprocessing as mp

import utils
import models.builer as builder
import dataloader

def get_args():
    # parse the args
    print('=> parse the args ...')
    parser = argparse.ArgumentParser(description='Trainer for auto encoder')
    parser.add_argument('--arch', default='vgg16', type=str, 
                        help='backbone architechture')
    parser.add_argument('--train_list', type=str)
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N', 
                        help='number of data loading workers (default: 0)')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                        'batch size of all GPUs on the current node when '
                        'using Data Parallel or Distributed Data Parallel')

    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', 
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')

    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 10)')

    parser.add_argument('--pth-save-fold', default='results/tmp', type=str,
                        help='The folder to save pths')
    parser.add_argument('--pth-save-epoch', default=1, type=int,
                        help='The epoch to save pth')
    parser.add_argument('--parallel', type=int, default=1, 
                        help='1 for parallel, 0 for non-parallel')
    parser.add_argument('--dist-url', default='tcp://localhost:10001', type=str,
                    help='url used to set up distributed training')
    parser.add_argument('--wandb_project_name', type=str, metavar='N',
                    help='wandb')
    parser.add_argument('--exp_name', type=str, metavar='N',
                    help='name of the experiment') 

    # add a store_true argument
    parser.add_argument('--leuven', action='store_true', help='leuven')
    # add a float argument 
    parser.add_argument('--lambda_', type=float, default=1, help='loss hyperparameter')
    
    args = parser.parse_args()

    return args

def setup_wandb(args):
    # use wandb api key
    wandb.login(key='18a861e71f78135d23eb672c08922edbfcb8d364')
    # start a wandb run
    id = wandb.util.generate_id()
    wandb.init(id = id, resume = "allow", project=args.wandb_project_name, entity="siddsuresh97") 
    config = wandb.config
    #name the wandb run
    wandb.run.name = args.exp_name
    print('=> wandb run name : {}'.format(wandb.run.name), args.exp_name)


def main(args):
    setup_wandb(args)
    print('=> torch version : {}'.format(torch.__version__))
    ngpus_per_node = torch.cuda.device_count()
    print('=> ngpus : {}'.format(ngpus_per_node))

    if args.parallel == 1: 
        # single machine multi card       
        args.gpus = ngpus_per_node
        args.nodes = 1
        args.nr = 0
        args.world_size = args.gpus * args.nodes

        args.workers = int(args.workers / args.world_size)
        args.batch_size = int(args.batch_size / args.world_size)
        mp.spawn(main_worker, nprocs=args.gpus, args=(args,))
    else:
        args.world_size = 1
        main_worker(1, args)

def LeuvenLoss(output, target, leuven_output, leuven_target, lambda_, iter, batch_size, pos_weights):
    # compute the loss which is a combination of the binary cross entropy loss and the mse loss
    # the loss is a weighted sum of the two losses
    # the weights are determined by the lambda parameter
    # compute the binary cross entropy loss
    # BCE = nn.BCEWithLogitsLoss(pos_weight=torch.from_numpy(pos_weights.to_numpy()).cuda(non_blocking=True))
    BCE = nn.BCEWithLogitsLoss() 
    # BCE = nn.BCELoss()
    # leuven_output = leuven_output.sigmoid() 
    BCE_loss = BCE(leuven_output, leuven_target.to(torch.float32))

    # compute the mse loss
    MSE = nn.MSELoss()
    MSE_loss = MSE(output, target)

    # # compute the total loss
    # if iter%30 == 0:
    #     print('BCE_loss: ', BCE_loss)
    #     print('MSE_loss: ', MSE_loss)
    total_loss = lambda_ * BCE_loss +  MSE_loss
    return total_loss,BCE_loss.item(), MSE_loss.item() 
    
def main_worker(gpu, args):
    # leuven_bce_transposed = pd.read_csv('leuven_bce_transposed_clean_with_pos_weight.csv', index_col=0)
    leuven_bce_transposed = pd.read_csv('leuven_bce_transposed.csv', index_col=0)
    utils.init_seeds(1 + gpu, cuda_deterministic=False)
    if args.parallel == 1:
        args.gpu = gpu
        args.rank = args.nr * args.gpus + args.gpu

        torch.cuda.set_device(gpu)
        torch.distributed.init_process_group(backend='nccl', init_method=args.dist_url, world_size=args.world_size, rank=args.rank)  
           
    else:
        # two dummy variable, not real
        args.rank = 0
        args.gpus = 1 
    if args.rank == 0:
        print('=> modeling the network {} ...'.format(args.arch))
    model = builder.BuildAutoEncoder(args) 
    if args.rank == 0:       
        total_params = sum(p.numel() for p in model.parameters())
        print('=> num of params: {} ({}M)'.format(total_params, int(total_params * 4 / (1024*1024))))
    
    if args.rank == 0:
        print('=> building the oprimizer ...')
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), args.lr,weight_decay=args.weight_decay)
    # optimizer = torch.optim.SGD(
    #         filter(lambda p: p.requires_grad, model.parameters()),
    #         args.lr,
    #         momentum=args.momentum,
    #         weight_decay=args.weight_decay)    
    if args.rank == 0:
        print('=> building the dataloader ...')
    train_loader = dataloader.train_loader(args)

    if args.rank == 0:
        print('=> building the criterion ...')
    
    criterion = nn.MSELoss()

    global iters
    iters = 0

    model.train()
    if args.rank == 0:
        print('=> starting training engine ...')
    for epoch in range(args.start_epoch, args.epochs):
        
        global current_lr
        current_lr = utils.adjust_learning_rate_cosine(optimizer, epoch, args)

        # train_loader.sampler.set_epoch(epoch)
        
        # train for one epoch
        do_train(train_loader, model, criterion, optimizer, epoch, args, leuven_bce_transposed)

        # save pth
        if epoch % args.pth_save_epoch == 0 and args.rank == 0:
            state_dict = model.state_dict()
            if not os.path.exists(os.path.join(args.pth_save_fold, args.exp_name)):
                os.makedirs(os.path.join(args.pth_save_fold, args.exp_name))
            torch.save(
                {
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': state_dict,
                    'optimizer' : optimizer.state_dict(),
                },
                os.path.join(args.pth_save_fold, args.exp_name, '{}.pth'.format(str(epoch).zfill(3)))
            )
            
            print(' : save pth for epoch {}'.format(epoch + 1))



def do_train(train_loader, model, criterion, optimizer, epoch, args, leuven_bce_transposed):
    leuven_list = leuven_bce_transposed.columns.to_list()
    leuven_list.sort()
    leuve_idx_to_class = {i: leuven_list[i] for i in range(len(leuven_list))}
    batch_time = utils.AverageMeter('Time', ':6.2f')
    data_time = utils.AverageMeter('Data', ':2.2f')
    losses = utils.AverageMeter('Loss', ':.4f')
    learning_rate = utils.AverageMeter('LR', ':.4f')
    
    progress = utils.ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, learning_rate],
        prefix="Epoch: [{}]".format(epoch+1))
    end = time.time()

    # update lr
    learning_rate.update(current_lr)

    for i, (input, target_img, target_class) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        global iters
        iters += 1 
        input = input.cuda(non_blocking=True)
        target_img = target_img.cuda(non_blocking=True) 
        if args.leuven:
            output, leuven_output = model(input)
            leuven_target = torch.tensor(np.array([leuven_bce_transposed[leuve_idx_to_class[i]].to_numpy() for i in target_class.cpu().numpy()])).cuda(non_blocking=True)
            leuven_target = leuven_target.cuda(non_blocking=True)
            # loss, bce_loss, mse_loss = LeuvenLoss(output, target_img, leuven_output, leuven_target, args.lambda_, iters, args.batch_size, leuven_bce_transposed['pos_weight'])
            loss, bce_loss, mse_loss = LeuvenLoss(output, target_img, leuven_output, leuven_target, args.lambda_, iters, args.batch_size, None)
        else:
            output = model(input)
            # the target is the input
            loss = criterion(output, input)
        # compute gradient and do solver step
        optimizer.zero_grad()
        # backward
        loss.backward()
        # update weights
        optimizer.step()

        # syn for logging
        torch.cuda.synchronize()

        # record loss
        losses.update(loss.item(), input.size(0))          

        # measure elapsed time
        if args.rank == 0:
            batch_time.update(time.time() - end)        
            end = time.time()   

        if i % args.print_freq == 0 and args.rank == 0:
            if args.leuven:
                wandb.log({"loss":loss.item(), "mse_loss":mse_loss, "bce_loss":bce_loss}, step=iters)
            else:
                wandb.log({"loss":loss.item()}, step=iters)
            progress.display(i)

if __name__ == '__main__':

    args = get_args()
    
    main(args)


