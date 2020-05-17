# %%
import gc
from typing import List

import torch
import argparse
import torch.optim as optim
from torch.optim.optimizer import Optimizer
import os
import sys
import shutil
import numpy as np

from torch.autograd import Variable
from torch.utils.data import DataLoader

from include.part2_skeleton.SGD import SGD
from include.part2_skeleton.dataset import FLICKR30K
from include.part2_skeleton.eval import mapk
from include.part2_skeleton.losses import cross_modal_hashing_loss
from include.part2_skeleton.models import BasicModel
from include.ranking import ranking

parser = argparse.ArgumentParser(description='Cross-modal Retrieval with Hashing')
parser.add_argument('--name', default='BasicModel', type=str,
                    help='name of experiment')
parser.add_argument('--gpu', type=int, default=0, metavar='N',
                    help='id of gpu to use')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--start_epoch', type=int, default=1, metavar='N',
                    help='number of start epoch (default: 1)')
parser.add_argument('--lr', type=float, default=3e-5, metavar='LR',
                    help='learning rate (default: 5e-5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--test', dest='test', action='store_true', default=False,
                    help='To only run inference on test set')
parser.add_argument('--dim_hidden', type=int, default=512, metavar='N',
                    help='how many hidden dimensions')
parser.add_argument('--c', type=int, default=32, metavar='N',
                    help='length of the binary code')
parser.add_argument('--margin', type=float, default=0.2, metavar='M',
                    help='margin in the loss function')
parser.add_argument('--gamma', type=float, default=1.0, metavar='M',
                    help='factor in the loss function')
parser.add_argument('--eta', type=float, default=1.0, metavar='M',
                    help='factor in the loss function')
parser.add_argument('--directory', default='include/output/model/hashing', type=str,
                    help='directory to save the best model')
parser.add_argument("--return_counts", type=bool, default=True)
parser.add_argument("--mode", default='client')
parser.add_argument("--port", default=52162)


# %%
def main():
    # enable GPU learning
    global args
    args = parser.parse_args()

    args.epochs = 20
    args.lr = 1e-3

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.cuda = False
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.set_device(args.gpu)
        print(torch.cuda.current_device())
        torch.cuda.manual_seed(args.seed)

    # obtain data loaders for train, validation and test sets
    train_set = FLICKR30K(mode='train', limit=5000)
    val_set = FLICKR30K(mode='val', limit=500)
    test_set = FLICKR30K(mode='test', limit=1000)

    print('datasets loaded')
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    print('loaders created')
    # create a model for cross-modal retrieval
    img_dim, txt_dim = train_set.get_dimensions()
    model = BasicModel(img_dim, txt_dim, args.dim_hidden, args.c)


    block = np.ones(5 ** 2).reshape(5, 5)
    # S = Variable(torch.from_numpy((np.kron(np.eye(len(train_set) // 5, dtype=int), block))))
    S = torch.from_numpy((np.kron(np.eye(len(train_set) // 5, dtype=int), block)))
    if args.cuda:
        model.cuda()

    best_map = 0

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            args.start_epoch = checkpoint['epoch'] + 1
            best_map = checkpoint['best_map']
            model.load_state_dict(checkpoint['state_dict'])
            model.to(torch.cuda.current_device())
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.test:
        test(test_loader, model)
        sys.exit()

    # set optimizer
    parameters = model.parameters()
    params = []
    for p in parameters:
        params.append(p), print(p.shape)
    print()
    optimizer = SGD(params, lr=args.lr)
    #optimizer = optim.SGD(parameters, lr=args.lr)
    #optimizer.add_param_group({'params': model.base.parameters()})
    n_parameters = sum([p.data.nelement() for p in model.parameters()])
    print('  + Number of params: {}'.format(n_parameters))

    map = test(val_loader, model, val_set.image_labels, val_set.caption_labels)

    # start training loop
    for epoch in range(args.start_epoch, args.epochs + 1):
        # update learning rate
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, S, optimizer, epoch)

        # evaluate on validation set
        map = test(val_loader, model, val_set.image_labels, val_set.caption_labels)

        # remember best MAP@10 and save checkpoint
        is_best = map > best_map
        best_map = max(map, best_map)
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_map': best_map,
        }, is_best)

    checkpoint = torch.load('runs/%s/' % (args.name) + 'model_best.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])
    test(test_loader, model, test_set.image_labels, test_set.caption_labels)

# %%
def train(train_loader, model, S, optimizer, epoch):
    losses = AverageMeter()
    maps = AverageMeter()

    # switch to train mode
    model.train()
    for batch_idx, (indices_x, x, indices_y, y) in enumerate(train_loader):

        gc.collect()

        if args.cuda:
            x = x.cuda()
            y = y.cuda()

        # pass data samples to model
        F, G, B = model.forward(x, y)
        indices_x = indices_x.type(torch.long)
        indices_y = indices_y.type(torch.long)
        # sim = get_similarity_matrix(indices_x, indices_y)
        sim = S[indices_x, indices_y]

        map_val = 0.0
        loss_value = cross_modal_hashing_loss(sim, F, G, B, 1, 1)

        # record MAP@10 and loss
        num_pairs = len(x)
        losses.update(loss_value.item(), num_pairs)
        maps.update(map_val, num_pairs)

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()

        # print progress
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{}]\t'
                  'Loss: {:.4f} ({:.4f}) \t'
                  'MAP@10: {:.2f}% ({:.2f}%)'.format(
                epoch, batch_idx * num_pairs, len(train_loader.dataset),
                losses.val, losses.avg,
                       100. * maps.val, 100. * maps.avg))


def test(test_loader, model, image_labels, caption_labels):
    # switch to evaluation mode
    model.eval()

    # TODO: Iterate over test set and evaluate MAP@10

    image_labels = image_labels.to_numpy()
    caption_labels = np.array(caption_labels)

    test_desc = []
    test_img = []

    for batch_idx, (x, y) in enumerate(test_loader):
        desc_embeddings, img_embeddings, _ = model(x, y)
        test_desc.append(np.transpose(desc_embeddings.detach().numpy()))
        test_img.append(np.transpose(img_embeddings.detach().numpy()))

    test_desc = np.concatenate(test_desc, axis=0)
    test_img = np.concatenate(test_img, axis=0)[::5]

    map_desc = mapk(test_desc, test_img, caption_labels, image_labels, 'desc')
    map_img = mapk(test_desc, test_img, caption_labels, image_labels, 'img')
    print('\n{} set: desc MAP@10: {:.2f}'.format(
        test_loader.dataset.mode,
        round(map_desc, 2)))
    print('\n{} set: desc MAP@10: {:.2f}'.format(
        test_loader.dataset.mode,
        round(map_img, 2)))

    map_avg = (map_desc + map_img) / 2

    print('{} set: MAP@10: {:.2f}\n'.format(
        test_loader.dataset.mode,
        round(map_avg, 2)))

    return map_avg

# %%
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """saves checkpoint to disk"""
    directory = args.directory
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'runs/%s/' % (args.name) + 'model_best.pth.tar')


class AverageMeter(object):
    """computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * ((1 - 0.015) ** epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# %%
if __name__ == '__main__':
    main()
