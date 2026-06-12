import argparse
import os
import random
import time
from enum import Enum
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim
import torch.utils.data
from torch.optim.lr_scheduler import StepLR
from utils.loader import get_training_data, get_validation_data

# Define arguments
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=256, type=int,
                    metavar='N')
parser.add_argument('--optimizer', default='adam', type=str,
                    help='Optimizer algorithm - "adam" for Adam optimizer, "sgd" for SGD optimizer.')
parser.add_argument('--lr', '--learning_rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--step_size', default=30, type=int,
                    help='step size (default: 30)')
parser.add_argument('--gamma', default=0.5, type=int,
                    help='gamma (default: 0.5)')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', default=False, dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('-freq', '--print_freq', default=1, type=int)
parser.add_argument('--vis_path', default='', type=str)
parser.add_argument('--log_features', default=False, type=bool)
parser.add_argument('--data', default='', metavar='DIR')
parser.add_argument('--num_samples', default=0, type=int)
parser.add_argument('--init_weight_path', default=None, type=str)
parser.add_argument('--quantile_regression', action='store_true')

# Type checking
args = parser.parse_args()

if args.quantile_regression:
    from QuantUNetT_model import QuantUNetT as PImodel
else:
    from unetT_model import UNetT as PImodel


def main():
    timestart = time.perf_counter()
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    cudnn.deterministic = True
    cudnn.benchmark = False

    main_worker(args.gpu, args)
    timeend = time.perf_counter()
    print("Total Time: %d" % (timeend - timestart))


class PinballLoss():
    def __init__(self, quantile=0.10, reduction='mean'):
        self.quantile = quantile
        assert 0 < self.quantile
        assert self.quantile < 1
        self.reduction = reduction

    def __call__(self, output, target):
        assert output.shape == target.shape
        loss = torch.zeros_like(target, dtype=torch.float)
        error = output - target
        smaller_index = error < 0
        bigger_index = 0 < error
        loss[smaller_index] = self.quantile * (abs(error)[smaller_index])
        loss[bigger_index] = (1 - self.quantile) * (abs(error)[bigger_index])

        if self.reduction == 'sum':
            loss = loss.sum()
        if self.reduction == 'mean':
            loss = loss.mean()

        return loss


class QuantileLoss(nn.Module):
    def __init__(self, q_lo=0.05, q_hi=0.95, q_lo_weight=1.0, q_hi_weight=1.0, mse_weight=1.0, reduction='mean'):
        super().__init__()
        self.q_lo = q_lo
        self.q_hi = q_hi
        self.q_lo_weight = q_lo_weight
        self.q_hi_weight = q_hi_weight
        self.mse_weight = mse_weight
        self.reduction = reduction

        self.q_lo_loss = PinballLoss(quantile=q_lo, reduction=reduction)
        self.q_hi_loss = PinballLoss(quantile=q_hi, reduction=reduction)
        self.mse_loss = nn.MSELoss(reduction=reduction)

    def forward(self, low, mu, high, target):
        q_lo_pred = low
        mean_pred = mu
        q_hi_pred = high

        loss = (
            self.q_lo_weight * self.q_lo_loss(q_lo_pred, target) +
            self.q_hi_weight * self.q_hi_loss(q_hi_pred, target) +
            self.mse_weight * self.mse_loss(mean_pred, target)
        )
        return loss


def main_worker(gpu, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    traindir = os.path.join(args.data)
    valdir = os.path.join(args.data)

    train_sampler = None
    val_sampler = None
    num_training_samples = args.num_samples
    train_dataset = get_training_data(traindir, num_training_samples)
    val_dataset = get_validation_data(valdir)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler)
    print(len(train_loader.dataset))
    print(len(val_loader.dataset))

    if torch.cuda.is_available():
        if args.gpu is not None:
            device = torch.device('cuda:{}'.format(args.gpu))
        else:
            device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    lr = args.lr
    momentum = args.momentum
    weight_decay = args.weight_decay

    model = PImodel()
    print(model)

    if args.init_weight_path is not None:
        init_state_dict = torch.load(args.init_weight_path, map_location='cpu')
        model_state_dict = model.state_dict()
    
        compatible_state_dict = {}
        skipped_keys = []
    
        for k, v in init_state_dict.items():
            if k in model_state_dict and model_state_dict[k].shape == v.shape:
                compatible_state_dict[k] = v
            else:
                skipped_keys.append(k)
    
        missing, unexpected = model.load_state_dict(compatible_state_dict, strict=False)
    
        print(f"Loaded weights from: {args.init_weight_path}")
        print(f"Successfully loaded keys: {len(compatible_state_dict)}")
        print(f"Skipped keys due to mismatch or missing: {len(skipped_keys)}")
        print(f"Missing keys after loading: {len(missing)}")
        print(f"Unexpected keys after loading: {len(unexpected)}")
    
        if len(skipped_keys) > 0:
            print("Skipped keys:")
            for k in skipped_keys:
                print(f"  {k}")

    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        print('using CPU, this will be slow')
    elif args.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        model = model.to(device)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        model = model.to(device)

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                     weight_decay=weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                                    weight_decay=weight_decay)

    if args.quantile_regression:
        criterion = QuantileLoss().to(device)
    else:
        criterion = nn.MSELoss().to(device)

    step_size = args.step_size
    gamma = args.gamma

    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    for epoch in range(args.start_epoch, args.epochs):
        train(train_loader, model, criterion, optimizer, epoch, device, args)
        scheduler.step()

    validate(val_loader, model, criterion, args, epoch, args.vis_path)


def train(train_loader, model, criterion, optimizer, epoch, device, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    model.train()

    end = time.time()
    for i, (I, Phi, fname) in enumerate(train_loader):
        data_time.update(time.time() - end)

        I_far = I.to(device, non_blocking=True)
        Phi = Phi.to(device, non_blocking=True)

        if args.quantile_regression:
            low, mu, high = model(I_far)
            loss = criterion(low, mu, high, Phi)
        else:
            output = model(I_far)
            loss = criterion(output, Phi)

        losses.update(loss.item(), Phi.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i + 1)


def validate(val_loader, model, criterion, args, epoch, output_dir):

    def run_validate(loader, epoch, output_dir, base_progress=0):
        inf_time_list = []
        avg = 0
        with torch.no_grad():
            end = time.time()
            num_batch = 0
            for x, (I, Phi) in enumerate(loader):
                x = base_progress + x

                if torch.cuda.is_available():
                    if args.gpu is not None:
                        I_far = I.cuda(args.gpu, non_blocking=True)
                        Phi = Phi.cuda(args.gpu, non_blocking=True)
                    else:
                        I_far = I.cuda(non_blocking=True)
                        Phi = Phi.cuda(non_blocking=True)
                elif torch.backends.mps.is_available():
                    I_far = I.to('mps')
                    Phi = Phi.to('mps')
                else:
                    I_far = I
                    Phi = Phi

                start = time.perf_counter()

                if args.quantile_regression:
                    low, output, high = model(I_far)
                    interval = high - low
                    interval = torch.clamp(interval, min=0.0)
                else:
                    output = model(I_far)
                    low = None
                    high = None
                    interval = torch.zeros_like(output)

                if output.dim() == 2:
                    output = output.unsqueeze(0)
                    Phi = Phi.unsqueeze(0)
                    I_far = I_far.unsqueeze(0)
                    interval = interval.unsqueeze(0)

                end_inf = time.perf_counter()
                inf_time_list.append(end_inf - start)
                num_batch += 1

                for i in range(len(output)):
                    outdata = output[i].squeeze().detach().cpu()
                    gtdata = Phi[i].squeeze().detach().cpu()

                    uncertainty = interval[i].squeeze().detach().cpu()

                    np.save(output_dir + '/Phi_pred/npy/' + str(x) + '_' + str(i) + '.npy', outdata)
                    plt.imsave(output_dir + '/Phi_pred/img/' + str(x) + '_' + str(i) + '.png', outdata, cmap='gray')
                    np.save(output_dir + '/Phi_gt/npy/' + str(x) + '_' + str(i) + '.npy', gtdata)
                    plt.imsave(output_dir + '/Phi_gt/img/' + str(x) + '_' + str(i) + '.png', gtdata, cmap='gray')

                    # np.save(output_dir+'/uncertainty/npy/'+str(x)+'_'+str(i)+'.npy', uncertainty)
                    # plt.imsave(output_dir+'/uncertainty/img/'+str(x)+'_'+str(i)+'.png' , uncertainty, cmap='gray')

                    GTdata = I_far[i].squeeze().detach().cpu()
                    np.save(output_dir + '/I_gt/npy/' + str(x) + '_' + str(i) + '.npy', GTdata)
                    plt.imsave(output_dir + '/I_gt/img/' + str(x) + '_' + str(i) + '.png', GTdata, cmap='gray')

                inf_time = np.array(inf_time_list)

                val_criterion = nn.MSELoss().to(Phi.device)
                loss = val_criterion(output, Phi)
                avg += loss.item()

                losses.update(loss.item(), Phi.size(0))

                batch_time.update(time.time() - end)
                end = time.time()

                progress.display(x + 1)

        return avg

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses],
        prefix='Test: ')

    model.eval()
    l = run_validate(val_loader, epoch, output_dir) / len(val_loader)
    progress.display_summary()

    return l


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
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

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count
        return self.avg

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


if __name__ == '__main__':
    main()