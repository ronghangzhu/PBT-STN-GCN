from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter

from metric import accuracy, roc_auc_compute_fn
from hyperparameter import *
from utils import load_citation, load_reddit_data
from models import *
from sample import Sampler
# from logger import Logger

# Training settings
parser = argparse.ArgumentParser()

# Tuning options
parser.add_argument('--tune_scales', '-tscl', action='store_true', default=True, help='whether to tune scales of perturbations by penalizing entropy on training set')
parser.add_argument('--tune_dropedge', '-tdrope', action='store_true', default=True, help='whether to tune edge dropout rates')
parser.add_argument('--tune_dropout', '-tdrop', action='store_true', default=True, help='whether to tune dropout rates (per layer)')
parser.add_argument('--tune_weightdecay', '-tweidec', action='store_true', default=True, help='whether to tune weight decay')
# Initial hyperparameter settings
parser.add_argument('--start_dropedge', '-drope', type=float, default=0.3, help='starting edge dropout rate')
parser.add_argument('--start_dropout', '-drop', type=float, default=0.05, help='starting dropout rate')
parser.add_argument('--start_weightdecay', '-weidec', type=float, default=5e-4, help='starting weightdecay rate')
# Optimization hyperparameters
parser.add_argument('--total_epochs', '-totep', type=int, default=400, help='number of training epochs to run for (warmup epochs are included in the count)')
parser.add_argument('--warmup_epochs', '-wupep', type=int, default=30, help='number of warmup epochs to run for before tuning hyperparameters')
parser.add_argument('--train_lr', '-tlr', type=float, default=5e-4, help='learning rate on parameters')
parser.add_argument('--valid_lr', '-vlr', type=float, default=3e-3, help='learning rate on hyperparameters')
parser.add_argument('--scale_lr', '-slr', type=float, default=1e-3, help='learning rate on scales (used if tuning scales)')
parser.add_argument('--momentum', '-mom', type=float, default=0.9, help='amount of momentum on usual parameters')
parser.add_argument('--train_steps', '-tstep', type=int, default=2, help='number of batches to optimize parameters on training set')
parser.add_argument('--valid_steps', '-vstep', type=int, default=1, help='number of batches to optimize hyperparameters on validation set')

# Regularization hyperparameters
parser.add_argument('--entropy_weight', '-ewt', type=float, default=1e-5, help='penalty applied to entropy of perturbation distribution')
parser.add_argument('--perturb_scale', '-pscl', type=float, default=0.5, help='scale of perturbation applied to continuous hyperparameters')

# Training parameter 
parser.add_argument('--no_cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument("--mixmode", action="store_true", default=False, help="Enable CPU GPU mixing mode.")
parser.add_argument('--dataset', default="citeseer", help="The data set")
parser.add_argument('--datapath', default="./data", help="The data path.")
parser.add_argument("--early_stopping", type=int, default=0, help="The patience of earlystopping. Do not adopt the earlystopping when it equals 0.")

# Model parameter
parser.add_argument('--hidden', type=int, default=128, help='Number of hidden units.')
parser.add_argument('--withbn', action='store_true', default=False, help='Enable Bath Norm GCN')
parser.add_argument('--withloop', action="store_true", default=False, help="Enable loop layer GCN")
parser.add_argument("--normalization", default="BingGeNormAdj", help="The normalization on the adj matrix.")
parser.add_argument("--task_type", default="full", help="The node classification task type (full and semi). Only valid for cora, citeseer and pubmed dataset.")

args = parser.parse_args()

os.environ["CUDA_VISIVLE_DEVICES"] = '0'

# pre setting
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.mixmode = args.no_cuda and args.mixmode and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

# random seed setting
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda or args.mixmode:
    torch.cuda.manual_seed(args.seed)

###############################################################################
# Data Loading/Processing
###############################################################################
sampler = Sampler(args.dataset, args.datapath, args.task_type)

# get labels and indexes
labels, idx_train, idx_val, idx_test = sampler.get_label_and_idxes(args.cuda)
nfeat = sampler.nfeat
nclass = sampler.nclass
print("nclass: %d\tnfea:%d" % (nclass, nfeat))

###############################################################################
# Model/Optimizer
###############################################################################
num_drops = 3  # this parameter decide the number of dropout
htensor, hscale, hdict = create_hparams(args, num_drops, device)

num_hparams = htensor.size()[0]
print('num_hparams is ', num_hparams)

# create hyper GCN model
model = GCN_H(nfeat=nfeat, nhid=args.hidden, nclass=nclass, num_hparams=num_hparams,
              activation=F.relu)
model = model.cuda()
total_params = sum(param.numel() for param in model.parameters())
print('Args: ', args)
print('total_params: ', total_params)

gcn_optimizer = optim.Adam(model.parameters(), lr=args.train_lr, weight_decay=args.start_weightdecay)
hyper_optimizer = optim.Adam([htensor], lr=args.valid_lr)
scale_optimizer = optim.Adam([hscale], lr=args.scale_lr)
scheduler = optim.lr_scheduler.MultiStepLR(gcn_optimizer, milestones=[200, 300, 400, 500, 600, 700], gamma=0.5)

###############################################################################
# Evaluation
###############################################################################
def evaluate(test_adj, test_fea, index, hyper=True):
    """
    return the loss and accuracy on the entire validation/test data
    """
    model.eval()

    hparam_tensor = hparam_transform(htensor.repeat(len(labels), 1), hdict)
    hnet_tensor = hnet_transform(htensor.repeat(len(labels), 1), hdict)

    output = model(test_fea, test_adj, hnet_tensor, hparam_tensor, hdict)
    loss_test = F.nll_loss(output[index], labels[index])
    acc_test = accuracy(output[index], labels[index])

    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          # "auc= {:.4f}".format(auc_test),
          "accuracy= {:.4f}".format(acc_test.item()))
    print("accuracy=%.5f" % (acc_test.item()))

    return loss_test, acc_test

###############################################################################
# Optimization step
###############################################################################
def optimization_step(sampler, index, hyper=False):
    if not hyper:
        model.train()
    else:
        model.eval()
    gcn_optimizer.zero_grad()
    hyper_optimizer.zero_grad()
    scale_optimizer.zero_grad()

    if not hyper or args.tune_scales:
        all_htensor = perturb(htensor, hscale, sampler.ndata)
    else:
        all_htensor = htensor.repeat(sampler.ndata, 1)

    hparam_tensor = hparam_transform(all_htensor, hdict)
    hnet_tensor = hnet_transform(all_htensor[:sampler.ndata], hdict)

    (train_adj, train_fea) = sampler.randomedge_sampler_hp(hparam_tensor, hdict,
                                                           normalization=args.normalization, cuda=args.cuda)
    output = model(train_fea, train_adj, hnet_tensor, hparam_tensor, hdict)
    loss_train = F.nll_loss(output[index], labels[index])
    acc_train = accuracy(output[index], labels[index])

    entropy = compute_entropy(hscale)
    loss = loss_train - args.entropy_weight * entropy
    loss.backward()

    if not hyper:
        gcn_optimizer.step()
        weight_decay = get_weightdecay_pro(hparam_tensor, hdict)
        gcn_optimizer.param_groups[0]['weight_decay'] = weight_decay.item()
    else:
        hyper_optimizer.step()
        if args.tune_scales:
            scale_optimizer.step()

    return loss_train, acc_train


def get_weightdecay_pro(hparam_tensor, hdict):
    if 'weightdecay' in hdict:
        drop_idx = hdict['weightdecay'].index
    else:
        print('No dropedge !!!!')
        return 0
    return hparam_tensor[0, drop_idx]


def get_dropedge_pro(hparam_tensor, hdict):
    if 'dropedge' in hdict:
        drop_idx = hdict['dropedge'].index
    else:
        print('No dropedge !!!!')
        return 0
    return hparam_tensor[:, drop_idx]
###############################################################################
# Training Loop
###############################################################################
train_step = valid_step = global_step = wup_step = 0
train_epoch = valid_epoch = 0
test_step = 0

#randomedge sampling
(train_adj, train_fea) = sampler.randomedge_sampler(percent=args.start_dropedge, normalization=args.normalization,
                                                    cuda=args.cuda)

model.train()
while train_epoch < args.warmup_epochs:
    optimization_step(sampler, idx_train)

    wup_step += 1
    global_step += 1
    train_epoch += 1

    (val_adj, val_fea) = sampler.get_test_set(normalization=args.normalization, cuda=args.cuda)
    val_loss, val_acc = evaluate(val_adj, val_fea, idx_val)

scheduler = optim.lr_scheduler.MultiStepLR(gcn_optimizer, milestones=[200, 300, 400, 500, 600, 700], gamma=0.5)

try:
    # Enter main training loop. Alternate between optimizing on training set for
    # args.train_steps and on validation set for args.valid_steps.
    while global_step < args.total_epochs:
        # check whether we should use training or validation set
        cycle_pos = (train_step + valid_step) % (args.train_steps + args.valid_steps)
        hyper = cycle_pos >= args.train_steps

        # do a step on the training set.
        if not hyper:
            model.train()
            train_loss, train_acc = optimization_step(sampler, idx_train, hyper)
            # train_loss, train_acc = optimization_step(train_epoch, train_adj, train_fea, idx_train)
            print('Tra_Epoch: {}  |  train_loss {:.4f}  |  train_acc {:.4f}'.format(global_step, train_loss, train_acc))
            train_step += 1
        # do a step on the validation set.
        else:
            model.eval()
            val_loss, val_acc = optimization_step(sampler, idx_val, hyper)
            # val_loss, val_acc = optimization_step(train_epoch, train_adj, train_fea, idx_val, hyper=True)
            print('Val_Epoch: {}  |  val_loss {:.4f}  |  val_acc {:.4f}'.format(global_step, val_loss, val_acc))
            valid_step += 1
        global_step += 1

except KeyboardInterrupt:
    print('=' * 80)
    print('Exiting from training early')


(test_adj, test_fea) = sampler.get_test_set(normalization=args.normalization, cuda=args.cuda)

# Run on val and test data.
val_loss, val_acc = evaluate(test_adj, test_fea, idx_val)
test_loss, test_acc = evaluate(test_adj, test_fea, idx_test)

print('=' * 89)
print('| End of training | val_loss {:8.5f} | val_acc {:8.5f} | test_loss {:8.5f} | test_acc {:8.5f}'.format(
         val_loss, val_acc, test_loss, test_acc))
print('=' * 89)




