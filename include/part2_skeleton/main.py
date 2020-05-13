# %%
import torch
import argparse
import torch.optim as optim
import os
import sys
import shutil
import numpy as np

from torch.autograd import Variable
from torch.utils.data import DataLoader

from part2_skeleton.dataset import FLICKR30K
from part2_skeleton.eval import mapk
from part2_skeleton.losses import cross_modal_hashing_loss
from part2_skeleton.models import BasicModel

parser = argparse.ArgumentParser(description='Cross-modal Retrieval with Hashing')
parser.add_argument('--name', default='BasicModel', type=str,
                    help='name of experiment')
parser.add_argument('--gpu', type=int, default=0, metavar='N',
                    help='id of gpu to use')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--start_epoch', type=int, default=1, metavar='N',
                    help='number of start epoch (default: 1)')
parser.add_argument('--lr', type=float, default=5e-5, metavar='LR',
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


# %%
def main():
    # enable GPU learning
    global args
    args = parser.parse_args()

    args.epochs = 100

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.cuda = False
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.set_device(args.gpu)
        print(torch.cuda.current_device())
        torch.cuda.manual_seed(args.seed)

    # obtain data loaders for train, validation and test sets
    train_set = FLICKR30K(mode='train', limit=1000)
    test_set = FLICKR30K(mode='test', limit=200)
    val_set = FLICKR30K(mode='val', limit=50)
    print('datasets loaded')
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    print('loaders created')
    # create a model for cross-modal retrieval
    img_dim, txt_dim = train_set.get_dimensions()
    model = BasicModel(len(train_set), img_dim, txt_dim, args.dim_hidden, args.c)

    block = np.ones(5 ** 2).reshape(5, 5)
    S = Variable(torch.from_numpy((np.kron(np.eye(len(train_set) // 5, dtype=int), block))))

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
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(parameters, lr=args.lr)
    n_parameters = sum([p.data.nelement() for p in model.parameters()])
    print('  + Number of params: {}'.format(n_parameters))

    # start training loop
    for epoch in range(args.start_epoch, args.epochs + 1):
        # update learning rate
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, S, optimizer, epoch)

        # evaluate on validation set
        map = test(val_loader, model)

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
    test(test_loader, model)


def train(train_loader, model, S, optimizer, epoch):
    losses = AverageMeter()
    maps = AverageMeter()

    # switch to train mode
    model.train()
    for batch_idx, (indices_x, x, indices_y, y) in enumerate(train_loader):
        if args.cuda:
            x = x.cuda()
            y = y.cuda()

        # pass data samples to model
        F, G, B = model(x, y)

        # sim = get_similarity_matrix(indices_x, indices_y)
        sim = S[indices_x, indices_y]

        # TODO: Use F, G and B to compute the MAP@10 and loss
        map = .1
        loss = cross_modal_hashing_loss(sim, F, G, B, 1, 1)

        # record MAP@10 and loss
        num_pairs = len(x)
        losses.update(loss.item(), num_pairs)
        maps.update(map, num_pairs)

        # compute gradient and do optimizer step
        optimizer.zero_grad()

        if loss == loss:
            loss.backward()
            optimizer.step()

        # print progress
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{}]\t'
                  'Loss: {:.4f} ({:.4f}) \t'
                  'MAP@10: {:.2f}% ({:.2f}%)'.format(
                epoch, batch_idx * num_pairs, len(train_loader.dataset),
                losses.val, losses.avg,
                       100. * maps.val, 100. * maps.avg))


def test(test_loader, model):
    # switch to evaluation mode
    model.eval()

    # TODO: Iterate over test set and evaluate MAP@10

    test_desc = []
    test_img = []
    #
    # for batch_idx, (x, y) in enumerate(test_loader):
    #     desc_embeddings, img_embeddings, _ = model(x, y)
    #     test_desc.append(desc_embeddings.detach().numpy())
    #     test_img.append(img_embeddings.detach().numpy())
    #
    # test_desc = np.stack(test_desc, axis=0)
    # test_img = np.stack(test_img, axis=0)

    map = mapk(test_desc, test_img)

    print('\n{} set: MAP@10: {:.2f}\n'.format(
        test_loader.dataset.mode,
        round(map * 100, 2)))

    return map


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """saves checkpoint to disk"""
    directory = "runs/%s/" % (args.name)
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


if __name__ == '__main__':
    main()
