import argparse
import torch
import numpy as np
import os.path as osp
import time
from data.mmot_data import MMotDataset
from models.gcn import GCN
from torch.utils.data import DataLoader
from utils.logging import Logger 
from utils.meters import AverageMeter
from utils.serialization import save_checkpoint
from utils.loss import My_loss
from utils.my_utils import my_collate_fn
import matplotlib.pyplot as plt
import os
from tensorboardX import SummaryWriter
import ipdb

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

writer = SummaryWriter('logs/1207')
steps = 0

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nusc_dataroot', type=str, default='/home/zhangxq/datasets/nuscenes')
    parser.add_argument('--info_path', type=str, default='datasets/ensemble_info/ensemble_info_train.json')
    parser.add_argument('--nusc_version', type=str, default='v1.0-trainval')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--logs_dir', type=str, default='logs/1212')
    parser.add_argument('--print_freq', type=int, default=1)
    args = parser.parse_args()
    
    return args

def train(loader, net, crit, opt, epoch, print_freq):
    losses = AverageMeter()

    net.train()
    for i, (node, adj, ids, scores) in enumerate(loader):
        node, adj = map(lambda x: x.cuda(), (node, adj))
        output, conf = net(node, adj)

        t =  node[:, :, 9]
        model_index = node[:, :, 12]
        loss = crit(output, conf, ids, scores, t, model_index)
        if loss > 10:
            ipdb.set_trace()
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        losses.update(loss.item(),node.size(0))
        
        if i % print_freq == 0:
            print('Epoch:[{0}][{1}/{2}]\t'
                  'Loss {losses.val:.3f} ({losses.avg:.3f})\t'.format(
                        epoch, i, len(loader), losses=losses))
        global steps
        writer.add_scalar('loss', loss, steps)
        steps += 1
        
def main():
    
    args = parse_config()
    
    # load data
    data = MMotDataset(args.info_path)
    
    trainloader = DataLoader(
        data, batch_size=args.batch_size, num_workers=args.workers, 
        shuffle=True, pin_memory=True, collate_fn=my_collate_fn)
    
    # data = MMotDataset(args.info_path)
    
    # valloader = DataLoader(data, batch_size=args.batch_size, shuffle=False)
    
    # init model
    net = GCN().cuda()
    
    # init optimizer and loss function
    opt = torch.optim.SGD(net.parameters(), args.lr, 
                          momentum=0.9, 
                          weight_decay=1e-4)
    # opt = torch.optim.AdamW(net.parameters(), lr=0.0001, 
    #                   betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)
    
    criterion = My_loss()
    
    # train
    loss_list = []
    for epoch in range(args.epochs):
        train(trainloader, net, criterion, opt, epoch, args.print_freq)
        # nodes = val(valloader, net, criterion, args.print_freq)
        if (epoch + 1) % 10 == 0:
            save_checkpoint({ 
                'state_dict':net.state_dict(),
                'epoch': epoch+1,}, False, 
                fpath=osp.join(args.logs_dir, 'epoch_{}.ckpt'.format(epoch + 1)))
    writer.close()
if __name__ == '__main__':
    main()