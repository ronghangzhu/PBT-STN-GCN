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

from earlystopping import EarlyStopping
from sample import Sampler
from metric import accuracy, roc_auc_compute_fn
# from deepgcn.utils import load_data, accuracy
# from deepgcn.models import GCN

from hyperparameter import *
from metric import accuracy
from utils import load_citation, load_reddit_data
from models import *
from earlystopping import EarlyStopping
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
parser.add_argument('--start_dropedge', '-drope', type=float, default=0.8, help='starting edge dropout rate')
parser.add_argument('--start_dropout', '-drop', type=float, default=0.1, help='starting dropout rate')
parser.add_argument('--start_weightdecay', '-weidec', type=float, default=5e-4, help='starting weightdecay rate')
# Optimization hyperparameters
parser.add_argument('--total_epochs', '-totep', type=int, default=4000, help='number of training epochs to run for (warmup epochs are included in the count)')
parser.add_argument('--warmup_epochs', '-wupep', type=int, default=0, help='number of warmup epochs to run for before tuning hyperparameters')
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
parser.add_argument('--dataset', default="cora", help="The data set")
parser.add_argument('--datapath', default="./data", help="The data path.")
parser.add_argument("--early_stopping", type=int, default=0, help="The patience of earlystopping. Do not adopt the earlystopping when it equals 0.")

# Model parameter
parser.add_argument('--hidden', type=int, default=128, help='Number of hidden units.')
parser.add_argument('--withbn', action='store_true', default=False, help='Enable Bath Norm GCN')
parser.add_argument('--withloop', action="store_true", default=False, help="Enable loop layer GCN")
parser.add_argument("--normalization", default="FirstOrderGCN", help="The normalization on the adj matrix.")
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
num_drops = 7  # this parameter decide the number of dropout
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
# Saving
###############################################################################
# SAVE
def save_text(data, name):
    np.savetxt(name, data)

dropout1 = []
dropout2 = []
dropout3 = []
dropout4 = []
dropout5 = []
dropout6 = []
dropout0 = []
dropedge = []
loss = []
acc = []
dict_save = {"dropout1": dropout1, "dropout2": dropout2, "dropout3": dropout3,
             "dropout4": dropout4, "dropout5": dropout5, "dropout6": dropout6,
             "dropout0": dropout0, "dropedge": dropedge, "loss": loss, "acc": acc}

###############################################################################
# Evaluation
###############################################################################
def evaluate(test_adj, test_fea, index, hyper=True):
    """
    return the loss and accuracy on the entire validation/test data
    """
    model.eval()

    if not hyper or args.tune_scales:
        all_htensor = perturb(htensor, hscale, sampler.ndata)
    else:
        all_htensor = htensor.repeat(sampler.ndata, 1)

    # hparam_tensor = hparam_transform(all_htensor, hdict)
    # hnet_tensor = hnet_transform(all_htensor[:sampler.ndata], hdict)

    hparam_tensor = hparam_transform(htensor.repeat(len(labels), 1), hdict)
    hnet_tensor = hnet_transform(htensor.repeat(len(labels), 1), hdict)

    output = model(test_fea, test_adj, hnet_tensor, hparam_tensor, hdict)
    # print('output is ', output)
    # output = model(test_fea, test_adj)
    loss_test = F.nll_loss(output[index], labels[index])
    acc_test = accuracy(output[index], labels[index])
    # auc_test = roc_auc_compute_fn(output[index], labels[index])

    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          # "auc= {:.4f}".format(auc_test),
          "accuracy= {:.4f}".format(acc_test.item()))
    print("accuracy=%.5f" % (acc_test.item()))

    return loss_test, acc_test

###############################################################################
# Optimization step
###############################################################################
# optimization_step(train_epoch, train_adj, train_fea, idx_train)
def optimization_step(sampler, index, dict_save, hyper=False):
    model.train()
    gcn_optimizer.zero_grad()
    hyper_optimizer.zero_grad()
    scale_optimizer.zero_grad()

    # print('hyper is ', hyper)
    # print('args.tune_scales is ', args.tune_scales)

    if not hyper or args.tune_scales:
        all_htensor = perturb(htensor, hscale, sampler.ndata)
    else:
        all_htensor = htensor.repeat(sampler.ndata, 1)

    hparam_tensor = hparam_transform(all_htensor, hdict)
    hnet_tensor = hnet_transform(all_htensor[:sampler.ndata], hdict)

    # zhu get hyparameters
    drop1_idx = hdict["dropout1"].index
    drop2_idx = hdict["dropout2"].index
    drop3_idx = hdict["dropout3"].index
    drop4_idx = hdict["dropout4"].index
    drop5_idx = hdict["dropout5"].index
    drop6_idx = hdict["dropout6"].index
    drop0_idx = hdict["dropout0"].index
    dropedge_idx = hdict["dropedge"].index

    dict_save["dropout1"].append(hparam_tensor[0, drop1_idx].cpu())
    dict_save["dropout2"].append(hparam_tensor[0, drop2_idx].cpu())
    dict_save["dropout3"].append(hparam_tensor[0, drop3_idx].cpu())
    dict_save["dropout4"].append(hparam_tensor[0, drop4_idx].cpu())
    dict_save["dropout5"].append(hparam_tensor[0, drop5_idx].cpu())
    dict_save["dropout6"].append(hparam_tensor[0, drop6_idx].cpu())
    dict_save["dropout0"].append(hparam_tensor[0, drop0_idx].cpu())
    dict_save["dropedge"].append(hparam_tensor[0, dropedge_idx].cpu())

    # print('drop edge is ', dropedge)
    # (train_adj, train_fea) = sampler.randomedge_sampler(percent=args.start_dropedge, normalization=args.normalization, cuda=args.cuda)
    (train_adj, train_fea) = sampler.randomedge_sampler_hp(hparam_tensor, hdict, normalization=args.normalization, cuda=args.cuda)

    output = model(train_fea, train_adj, hnet_tensor, hparam_tensor, hdict)
    loss_train = F.nll_loss(output[index], labels[index])
    acc_train = accuracy(output[index], labels[index])

    # copy from the STN
    entropy = compute_entropy(hscale)
    loss = loss_train - args.entropy_weight * entropy
    loss.backward()

    if not hyper:
        gcn_optimizer.step()
        weight_decay = get_weightdecay_pro(hparam_tensor, hdict)
        # print("gcn_optimizer.param_groups is ", gcn_optimizer.param_groups[0]["weight_decay"])
        gcn_optimizer.param_groups[0]['weight_decay'] = weight_decay.item()
        # print("gcn_optimizer.param_groups[0]['weight_decay'] is ", gcn_optimizer.param_groups[0]['weight_decay'])
    else:
        hyper_optimizer.step()
        if args.tune_scales:
            scale_optimizer.step()

    with torch.no_grad():
        (test_adj, test_fea) = sampler.get_test_set(normalization=args.normalization, cuda=args.cuda)
        loss_test, acc_test = evaluate(test_adj, test_fea, idx_test)
        dict_save["loss"].append(loss_test.cpu())
        dict_save["acc"].append(acc_test.cpu())

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
# Bookkeeping stuff.
train_step = valid_step = global_step = wup_step = 0
train_epoch = valid_epoch = 0
test_step = 0

# need use the edge
#randomedge sampling

(train_adj, train_fea) = sampler.randomedge_sampler(percent=args.start_dropedge, normalization=args.normalization,
                                                    cuda=args.cuda)

model.train()
while train_epoch < args.warmup_epochs:
    # def optimization_step(epoch, sampler, idx, hyper=False):
    optimization_step(sampler, idx_train)

    wup_step += 1
    global_step += 1
    train_epoch += 1

    (val_adj, val_fea) = sampler.get_test_set(normalization=args.normalization, cuda=args.cuda)
    val_loss, val_acc = evaluate(val_adj, val_fea, idx_val)
    print('val_loss is ', val_loss)
    # print('=' * 80)
    # print('Warm Epoch: {}  |  val_loss{:.4f}  |  val_acc{.4f}'.format(wup_step, val_loss, val_acc))
    # print('=' * 80)


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
            train_loss, train_acc = optimization_step(sampler, idx_train, dict_save, hyper)
            # train_loss, train_acc = optimization_step(train_epoch, train_adj, train_fea, idx_train)
            print('Tra_Epoch: {}  |  train_loss{:.4f}  |  train_acc{:.4f}'.format(global_step, train_loss, train_acc))
            train_step += 1
        # do a step on the validation set.
        else:
            model.eval()
            val_loss, val_acc = optimization_step(sampler, idx_val, dict_save, hyper)
            # val_loss, val_acc = optimization_step(train_epoch, train_adj, train_fea, idx_val, hyper=True)
            print('Val_Epoch: {}  |  val_loss{:.4f}  |  val_acc{:.4f}'.format(global_step, val_loss, val_acc))
            valid_step += 1

        global_step += 1

except KeyboardInterrupt:
    print('=' * 80)
    print('Exiting from training early')



save_text(dict_save["dropout1"], "./text/Cora_Sgcn8_dropout1.txt")
save_text(dict_save["dropout2"], "./text/Cora_Sgcn8_dropout2.txt")
save_text(dict_save["dropout3"], "./text/Cora_Sgcn8_dropout3.txt")
save_text(dict_save["dropout4"], "./text/Cora_Sgcn8_dropout4.txt")
save_text(dict_save["dropout5"], "./text/Cora_Sgcn8_dropout5.txt")
save_text(dict_save["dropout6"], "./text/Cora_Sgcn8_dropout6.txt")
save_text(dict_save["dropout0"], "./text/Cora_Sgcn8_dropout0.txt")
save_text(dict_save["dropedge"], "./text/Cora_Sgcn8_dropedge.txt")
save_text(dict_save["loss"], "./text/Cora_Sgcn8_loss.txt")
save_text(dict_save["acc"], "./text/Cora_Sgcn8_acc.txt")

(test_adj, test_fea) = sampler.get_test_set(normalization=args.normalization, cuda=args.cuda)

# Run on val and test data.
val_loss, val_acc = evaluate(test_adj, test_fea, idx_val)
test_loss, test_acc = evaluate(test_adj, test_fea, idx_test)

print('=' * 89)
print('| End of training | val_loss {:8.5f} | val_acc {:8.5f} | test_loss {:8.5f} | test_acc {:8.5f}'.format(
         val_loss, val_acc, test_loss, test_acc))
print('=' * 89)











#
#
#
# # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, factor=0.618)
# scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200, 300, 400, 500, 600, 700], gamma=0.5)
# # convert to cuda
# if args.cuda:
#     model.cuda()
#
# # For the mix mode, lables and indexes are in cuda.
# if args.cuda or args.mixmode:
#     labels = labels.cuda()
#     idx_train = idx_train.cuda()
#     idx_val = idx_val.cuda()
#     idx_test = idx_test.cuda()
#
# if args.warm_start is not None and args.warm_start != "":
#     early_stopping = EarlyStopping(fname=args.warm_start, verbose=False)
#     print("Restore checkpoint from %s" % (early_stopping.fname))
#     model.load_state_dict(early_stopping.load_checkpoint())
#
# # set early_stopping
# if args.early_stopping > 0:
#     early_stopping = EarlyStopping(patience=args.early_stopping, verbose=False)
#     print("Model is saving to: %s" % (early_stopping.fname))
#
# if args.no_tensorboard is False:
#     tb_writer = SummaryWriter(
#         comment=f"-dataset_{args.dataset}-type_{args.type}"
#     )
#
# def get_lr(optimizer):
#     for param_group in optimizer.param_groups:
#         return param_group['lr']
#
#
# # define the training function.
# def train(epoch, train_adj, train_fea, idx_train, val_adj=None, val_fea=None):
#     if val_adj is None:
#         val_adj = train_adj
#         val_fea = train_fea
#
#     t = time.time()
#     model.train()
#     optimizer.zero_grad()
#     output = model(train_fea, train_adj)
#     # special for reddit
#     if sampler.learning_type == "inductive":
#         loss_train = F.nll_loss(output, labels[idx_train])
#         acc_train = accuracy(output, labels[idx_train])
#     else:
#         loss_train = F.nll_loss(output[idx_train], labels[idx_train])
#         acc_train = accuracy(output[idx_train], labels[idx_train])
#
#     loss_train.backward()
#     optimizer.step()
#     train_t = time.time() - t
#     val_t = time.time()
#     # We can not apply the fastmode for the reddit dataset.
#     # if sampler.learning_type == "inductive" or not args.fastmode:
#
#     if args.early_stopping > 0 and sampler.dataset != "reddit":
#         loss_val = F.nll_loss(output[idx_val], labels[idx_val]).item()
#         early_stopping(loss_val, model)
#
#     if not args.fastmode:
#         #    # Evaluate validation set performance separately,
#         #    # deactivates dropout during validation run.
#         model.eval()
#         output = model(val_fea, val_adj)
#         loss_val = F.nll_loss(output[idx_val], labels[idx_val]).item()
#         acc_val = accuracy(output[idx_val], labels[idx_val]).item()
#         if sampler.dataset == "reddit":
#             early_stopping(loss_val, model)
#     else:
#         loss_val = 0
#         acc_val = 0
#
#     if args.lradjust:
#         scheduler.step()
#
#     val_t = time.time() - val_t
#     return (loss_train.item(), acc_train.item(), loss_val, acc_val, get_lr(optimizer), train_t, val_t)
#
#
# def test(test_adj, test_fea):
#     model.eval()
#     output = model(test_fea, test_adj)
#     loss_test = F.nll_loss(output[idx_test], labels[idx_test])
#     acc_test = accuracy(output[idx_test], labels[idx_test])
#     auc_test = roc_auc_compute_fn(output[idx_test], labels[idx_test])
#     if args.debug:
#         print("Test set results:",
#               "loss= {:.4f}".format(loss_test.item()),
#               "auc= {:.4f}".format(auc_test),
#               "accuracy= {:.4f}".format(acc_test.item()))
#         print("accuracy=%.5f" % (acc_test.item()))
#     return (loss_test.item(), acc_test.item())
#
#
# # Train model
# t_total = time.time()
# loss_train = np.zeros((args.epochs,))
# acc_train = np.zeros((args.epochs,))
# loss_val = np.zeros((args.epochs,))
# acc_val = np.zeros((args.epochs,))
#
# sampling_t = 0
#
# for epoch in range(args.epochs):
#     input_idx_train = idx_train
#     sampling_t = time.time()
#     # no sampling
#     # randomedge sampling if args.sampling_percent >= 1.0, it behaves the same as stub_sampler.
#     (train_adj, train_fea) = sampler.randomedge_sampler(percent=args.sampling_percent, normalization=args.normalization,
#                                                         cuda=args.cuda)
#     if args.mixmode:
#         train_adj = train_adj.cuda()
#
#     sampling_t = time.time() - sampling_t
#
#     # The validation set is controlled by idx_val
#     # if sampler.learning_type == "transductive":
#     if False:
#         outputs = train(epoch, train_adj, train_fea, input_idx_train)
#     else:
#         (val_adj, val_fea) = sampler.get_test_set(normalization=args.normalization, cuda=args.cuda)
#         if args.mixmode:
#             val_adj = val_adj.cuda()
#         outputs = train(epoch, train_adj, train_fea, input_idx_train, val_adj, val_fea)
#
#     if args.debug and epoch % 1 == 0:
#         print('Epoch: {:04d}'.format(epoch + 1),
#               'loss_train: {:.4f}'.format(outputs[0]),
#               'acc_train: {:.4f}'.format(outputs[1]),
#               'loss_val: {:.4f}'.format(outputs[2]),
#               'acc_val: {:.4f}'.format(outputs[3]),
#               'cur_lr: {:.5f}'.format(outputs[4]),
#               's_time: {:.4f}s'.format(sampling_t),
#               't_time: {:.4f}s'.format(outputs[5]),
#               'v_time: {:.4f}s'.format(outputs[6]))
#
#     if args.no_tensorboard is False:
#         tb_writer.add_scalars('Loss', {'train': outputs[0], 'val': outputs[2]}, epoch)
#         tb_writer.add_scalars('Accuracy', {'train': outputs[1], 'val': outputs[3]}, epoch)
#         tb_writer.add_scalar('lr', outputs[4], epoch)
#         tb_writer.add_scalars('Time', {'train': outputs[5], 'val': outputs[6]}, epoch)
#
#
#     loss_train[epoch], acc_train[epoch], loss_val[epoch], acc_val[epoch] = outputs[0], outputs[1], outputs[2], outputs[
#         3]
#
#     if args.early_stopping > 0 and early_stopping.early_stop:
#         print("Early stopping.")
#         model.load_state_dict(early_stopping.load_checkpoint())
#         break
#
# if args.early_stopping > 0:
#     model.load_state_dict(early_stopping.load_checkpoint())
#
# if args.debug:
#     print("Optimization Finished!")
#     print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
#
# # Testing
# (test_adj, test_fea) = sampler.get_test_set(normalization=args.normalization, cuda=args.cuda)
# if args.mixmode:
#     test_adj = test_adj.cuda()
# (loss_test, acc_test) = test(test_adj, test_fea)
# print("%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f" % (
# loss_train[-1], loss_val[-1], loss_test, acc_train[-1], acc_val[-1], acc_test))
