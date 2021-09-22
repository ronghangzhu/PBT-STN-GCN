import math
import sys
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


###############################################################################
# Miscellaneous functions
###############################################################################

def logit(x):
    return torch.log(x) - torch.log(1-x)

def s_logit(x, min=0, max=1):
    """Stretched logit function: Maps x lying in (min, max) to R"""
    return logit((x - min)/(max-min))

def s_sigmoid(x, min=0, max=1):
    """Stretched sigmoid function: Maps x lying in R to (min, max)"""
    return (max-min)*torch.sigmoid(x) + min

def inv_softplus(x):
    """ Inverse softplus function: Maps x lying in (0, infty) to R"""
    return torch.log(torch.exp(x) - 1)

def robustify(x, eps):
    """
    Adjusts x lying in an interval [a, b] so that it lies in [a+eps, b-eps]
    through a linear projection to that interval.
    """
    return (1-2*eps) * x + eps

# hparam = project(hvalue, range_min, range_max)
def project(x, range_min, range_max):
    if range_min == -float('inf') and range_max == float('inf'):
        return x
    elif range_min == -float('inf') and range_max != float('inf'):
        return range_max - F.softplus(x)
    elif range_min != -float('inf') and range_max == float('inf'):
        return range_min + F.softplus(x)
    elif range_min != -float('inf') and range_max != float('inf'):
        return s_sigmoid(x, range_min, range_max)

def gaussian_cdf(x):
    """
    Computes cdf of standard normal distribution.

    Arguments:
        x (Tensor)
    """
    return 0.5 * (1 + torch.erf(x / math.sqrt(2)))

###############################################################################
# Hyperparameter convenience functions
###############################################################################
class HyperparameterInfo():
    def __init__(self, index, range, hnet_fcn, discrete=False, minibatch=True):
        """
        Arguments:
            index (int): index of hyperparameter in htensor
            range (float, float): tuple specifying range hyperparameter must lie in
            hnet_fcn (fcn): function applied to unconstrained hyperparameter before it's fed into the hypernet
            discrete (bool): whether hyperparameter is discrete
            minibatch (bool): whether hyperparameter can use minibatched perturbations
        """
        self.index = index
        self.range = range
        self.hnet_fcn = hnet_fcn
        self.discrete = discrete
        self.minibatch = minibatch

def perturb(htensor, hscale, batch_size, hdict=None):
    """
    Arguments:
        htensor (tensor): size is (H,)
        hscale (tensor): size is (H,)

    Returns:
        perturb_htensor (tensor): perturbed hyperparameters in unconstrained space,
            size is (B, H)
    """
    noise = htensor.new(batch_size, htensor.size(0)).normal_()

    perturb_htensor = htensor + F.softplus(hscale)*noise
    if hdict is not None:
        for hinfo in [hinfo for hinfo in hdict.values() if not hinfo.minibatch]:
            hidx = hinfo.index
            perturb_htensor[:, hidx] = perturb_htensor[0, hidx]
    return perturb_htensor

def hnet_transform(htensor, hdict):
    """
    Arguments:
        htensor (tensor): size is (B, H)
        hdict: dictionary mapping hyperparameter names to relevant info

    Returns:
        hnet_tensor (tensor): tensor of size (B, H) ready to be fed into hypernet
    """
    hnet_tensor_list = []
    for hinfo in hdict.values():
        hvalue = htensor[:,hinfo.index]
        hnet_tensor_list.append(hinfo.hnet_fcn(hvalue))
    return torch.stack(hnet_tensor_list, dim=1)

def compute_entropy(hscale):
    """
    Arguments:
        hscale (tensor): size is (H,)

    Returns:
        entropy (tensor): returns scalar value of entropy of perturbation distribution
    """
    scale = F.softplus(hscale)
    return torch.sum(torch.log(scale * math.sqrt(2*math.pi*math.e)))

def hparam_transform(htensor, hdict):
    """
    Arguments:
        htensor (tensor): size is (B, H)
        hdict: dictionary mapping hyperparameter names to relevant info

    Returns:
        hparam_tensor (tensor): tensor ready to be used as actual hyperparameters
    """
    hparam_tensor_list = []
    for hinfo in hdict.values():
        range_min, range_max = hinfo.range
        hvalue = htensor[:,hinfo.index]
        hparam = project(hvalue, range_min, range_max)

        if hinfo.discrete:
            hparam = torch.floor(hparam)
        hparam_tensor_list.append(hparam)

    # zhu: check size
    # print('the size of hvalue is ', hvalue.size())
    # print('the size of htensor is ', htensor.size())
    # print('the len of hparam_tensor_list is ', len(hparam_tensor_list))
    return torch.stack(hparam_tensor_list, dim=1)

def create_hlabels(hdict, args):
    """Returns a tuple of the names of hyperparameters being tuned and their scales
    (if they are being tuned)."""
    hlabels = list(hdict.keys())
    if 'tune_scales' in args and args.tune_scales:
        hlabels += [hlabel + '_scale' for hlabel in hlabels]
    hlabels = tuple(hlabels)
    return hlabels

def create_hstats(htensor, hscale, hdict, args):
    """Returns a dictionary mapping names of hyperparameters to their current
    values.
    """
    hstats = OrderedDict()

    for hname, hinfo in hdict.items():
        range_min, range_max = hinfo.range
        hstats[hname] = project(htensor[hinfo.index], range_min, range_max).item()
        if 'tune_scales' in args and args.tune_scales:
            hstats[hname + '_scale'] = F.softplus(hscale[hinfo.index]).item()
    return hstats

###############################################################################
# create htensor hscale and hdict
###############################################################################

def create_hparams(args, num_drops, device):
    """
    Arguments:
        args: the arguments supplied by the user to the main script
        cnn_class: the convolutional net class
        device: device we are training on
    Returns:
        htensor: unconstrained reparametrization of the starting hyperparameters
        hscale: unconstrained reparametrization of the perturbation distribution's scale
        hdict: dictionary mapping hyperparameter names to info about the hyperparameter
    """
    hdict = OrderedDict()
    htensor_list = []
    hscale_list = []

    drop_max = 0.9 # improves stability
    if args.tune_dropedge:
        htensor_list.append(s_logit(torch.tensor(args.start_dropedge), min=0.1, max=1))
        hscale_list.append(inv_softplus(torch.tensor(args.perturb_scale)))
        hdict['dropedge'] = HyperparameterInfo(index=len(hdict), range=(0.1, 1), hnet_fcn=lambda x: x)

    if args.tune_dropout:
        for i in range(num_drops):
            htensor_list.append(s_logit(torch.tensor(args.start_dropout), min=0., max=0.9))
            hscale_list.append(inv_softplus(torch.tensor(args.perturb_scale)))
            hdict['dropout' + str(i)] = HyperparameterInfo(index=len(hdict), range=(0., 0.9), hnet_fcn=lambda x: x)

    if args.tune_weightdecay:
        htensor_list.append(s_logit(torch.tensor(args.start_weightdecay), min=1e-6, max=1e-2))
        hscale_list.append(inv_softplus(torch.tensor(args.perturb_scale)))
        hdict['weightdecay'] = HyperparameterInfo(index=len(hdict), range=(1e-6, 1e-2), hnet_fcn=lambda x: x)


    htensor = nn.Parameter(torch.stack(htensor_list).to(device))
    hscale = nn.Parameter(torch.stack(hscale_list).to(device))
    return htensor, hscale, hdict
