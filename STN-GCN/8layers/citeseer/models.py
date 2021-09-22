import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import *
from torch.nn.parameter import Parameter

device = torch.device("cuda:0")



class GCN_H(nn.Module):
    """
       The model for the single kind of deepgcn blocks.

       The model architecture likes:
       inputlayer(nfeat)--block(nbaselayer, nhid)--...--outputlayer(nclass)--softmax(nclass)
                           |------  nhidlayer  ----|
       The total layer is nhidlayer*nbaselayer + 2.
       All options are configurable.
    """
    def __init__(self, nfeat, nhid, nclass, num_hparams,
                 activation=lambda x: x, mixmode=False):
        super(GCN_H, self).__init__()
        self.mixmode = mixmode
        # self.dropout = dropout
        self.hiddenlayers = nn.ModuleList()

        self.ingc = GraphConvolutionBS_Hyper(nfeat, nhid, num_hparams, activation)
        # self.ingc = GraphConvolutionBS(nfeat, nhid, activation, withbn, withloop)
        baseblockinput = nhid
        outactivation = lambda x: x
        self.outgc = GraphConvolutionBS_Hyper(baseblockinput, nclass, num_hparams, outactivation)

        self.hid_layer1 = GraphConvolutionBS_Hyper(nhid, nhid, num_hparams, activation)
        self.hid_layer2 = GraphConvolutionBS_Hyper(nhid, nhid, num_hparams, activation)
        self.hid_layer3 = GraphConvolutionBS_Hyper(nhid, nhid, num_hparams, activation)
        self.hid_layer4 = GraphConvolutionBS_Hyper(nhid, nhid, num_hparams, activation)
        self.hid_layer5 = GraphConvolutionBS_Hyper(nhid, nhid, num_hparams, activation)
        self.hid_layer6 = GraphConvolutionBS_Hyper(nhid, nhid, num_hparams, activation)

        # for i in range(nhidlayer):
        #     layer = GraphConvolutionBS_Hyper(activation, self.withbn, self.withloop,in_features=nhid, out_features=nhid)
        #     self.hiddenlayers.append(layer)

        # self.midlayer = nn.ModuleList()
        # for i in range(nhidlayer):
        #     gcb = MultiLayerGCNBlock(in_features=baseblockinput,
        #                          out_features=nhid,
        #                          nbaselayer=nbaselayer,
        #                          withbn=withbn,
        #                          withloop=withloop,
        #                          activation=activation,
        #                          dropout=dropout,
        #                          dense=False,
        #                          aggrmethod=aggrmethod)
        #     self.midlayer.append(gcb)
        #     baseblockinput = gcb.get_outdim()

        # outactivation = lambda x: x  # we donot need nonlinear activation here.
        # self.outgc = GraphConvolutionBS(nhid, nclass, outactivation, withbn, withloop)
        if mixmode:
            self.ingc = self.ingc.to(device)
            self.hid_layer1 = self.hid_layer1.to(device)
            self.hid_layer2 = self.hid_layer2.to(device)
            self.hid_layer3 = self.hid_layer3.to(device)
            self.hid_layer4 = self.hid_layer4.to(device)
            self.hid_layer5 = self.hid_layer5.to(device)
            self.hid_layer6 = self.hid_layer6.to(device)
            # self.hiddenlayers = self.hiddenlayers(device)
            self.outgc = self.outgc.to(device)

    def forward(self, fea, adj, hnet_tensor, hparam_tensor, hdict):
        x = self.ingc(fea, adj, hnet_tensor)
        drop_probs = self.get_drop_probs(hparam_tensor, hdict, 0)  # keywords indict the index in Hparams
        x = self.dropout(x, drop_probs, training=self.training)

        x = self.hid_layer1(x, adj, hnet_tensor)
        drop_probs = self.get_drop_probs(hparam_tensor, hdict, 1)
        x = self.dropout(x, drop_probs, training=self.training)

        x = self.hid_layer2(x, adj, hnet_tensor)
        drop_probs = self.get_drop_probs(hparam_tensor, hdict, 2)
        x = self.dropout(x, drop_probs, training=self.training)

        x = self.hid_layer3(x, adj, hnet_tensor)
        drop_probs = self.get_drop_probs(hparam_tensor, hdict, 3)
        x = self.dropout(x, drop_probs, training=self.training)

        x = self.hid_layer4(x, adj, hnet_tensor)
        drop_probs = self.get_drop_probs(hparam_tensor, hdict, 4)
        x = self.dropout(x, drop_probs, training=self.training)

        x = self.hid_layer5(x, adj, hnet_tensor)
        drop_probs = self.get_drop_probs(hparam_tensor, hdict, 5)
        x = self.dropout(x, drop_probs, training=self.training)

        x = self.hid_layer6(x, adj, hnet_tensor)
        drop_probs = self.get_drop_probs(hparam_tensor, hdict, 6)
        x = self.dropout(x, drop_probs, training=self.training)

        x = self.outgc(x, adj, hnet_tensor)
        x = F.log_softmax(x, dim=1)

        return x

    def get_drop_probs(self, hparam_tensor, hdict, keyword):
        if 'dropout' + str(keyword) in hdict:
            drop_idx = hdict['dropout' + str(keyword)].index
            return hparam_tensor[:, drop_idx]
        else:
            print('Can not get the dropout probs!!!!')
            return 0
        #
        # if 'dropout' in hdict:
        #     drop_idx = hdict['dropout'].index
        # elif 'dropout' + str(layer) in hdict:
        #     drop_idx = hdict['dropout' + str(layer)].index
        # else:
        #     return 0.
        # return hparam_tensor[:, drop_idx]

    def get_fcdrop_probs(self, hparam_tensor, hdict):
        if 'fcdropout0' not in hdict:
            return (0., 0.)
        fcdrop0_idx = hdict['fcdropout0'].index
        fcdrop1_idx = hdict['fcdropout1'].index
        return (hparam_tensor[:,fcdrop0_idx], hparam_tensor[:, fcdrop1_idx])

    # def dropout(self, x, probs, training=False):
    #     """
    #     Arguments:
    #         x (Tensor): whose first dimension has size B
    #         probs (Tensor): size (B,)
    #     """
    #     prob = probs[0]
    #     if not training:
    #         return x
    #     if isinstance(probs, float):
    #         return F.dropout(x, prob, training)
    #     x_size = x.size()
    #     x = x.view(x.size(0), -1)
    #     probs = probs.unsqueeze(1).repeat(1, x.size(1)).detach()
    #     mask = (1 - probs).bernoulli().div_(1 - probs)
    #     return (x * mask).view(x_size)

    def dropout(self, x, probs, training=False):
        """
        Arguments:
            x (Tensor): whose first dimension has size B
            probs (Tensor): size (B,)
        """
        if not training:
            return x
        if isinstance(probs, float):
            return F.dropout(x, probs, training)
        x_size = x.size()
        x = x.view(x.size(0), -1)
        probs = probs.unsqueeze(1).repeat(1, x.size(1)).detach()
        mask = (1 - probs).bernoulli().div_(1 - probs)
        return (x * mask).view(x_size)


class GCNModel(nn.Module):
    """
       The model for the single kind of deepgcn blocks.

       The model architecture likes:
       inputlayer(nfeat)--block(nbaselayer, nhid)--...--outputlayer(nclass)--softmax(nclass)
                           |------  nhidlayer  ----|
       The total layer is nhidlayer*nbaselayer + 2.
       All options are configurable.
    """

    def __init__(self,
                 nfeat,
                 nhid,
                 nclass,
                 nhidlayer,
                 dropout,
                 baseblock="mutigcn",
                 inputlayer="gcn",
                 outputlayer="gcn",
                 nbaselayer=0,
                 activation=lambda x: x,
                 withbn=True,
                 withloop=True,
                 aggrmethod="add",
                 mixmode=False):
        """
        Initial function.
        :param nfeat: the input feature dimension.
        :param nhid:  the hidden feature dimension.
        :param nclass: the output feature dimension.
        :param nhidlayer: the number of hidden blocks.
        :param dropout:  the dropout ratio.
        :param baseblock: the baseblock type, can be "mutigcn", "resgcn", "densegcn" and "inceptiongcn".
        :param inputlayer: the input layer type, can be "gcn", "dense", "none".
        :param outputlayer: the input layer type, can be "gcn", "dense".
        :param nbaselayer: the number of layers in one hidden block.
        :param activation: the activation function, default is ReLu.
        :param withbn: using batch normalization in graph convolution.
        :param withloop: using self feature modeling in graph convolution.
        :param aggrmethod: the aggregation function for baseblock, can be "concat" and "add". For "resgcn", the default
                           is "add", for others the default is "concat".
        :param mixmode: enable cpu-gpu mix mode. If true, put the inputlayer to cpu.
        """
        super(GCNModel, self).__init__()
        self.mixmode = mixmode
        self.dropout = dropout

        if baseblock == "resgcn":
            self.BASEBLOCK = ResGCNBlock
        elif baseblock == "densegcn":
            self.BASEBLOCK = DenseGCNBlock
        elif baseblock == "mutigcn":
            self.BASEBLOCK = MultiLayerGCNBlock
        elif baseblock == "inceptiongcn":
            self.BASEBLOCK = InecptionGCNBlock
        else:
            raise NotImplementedError("Current baseblock %s is not supported." % (baseblock))
        if inputlayer == "gcn":
            # input gc
            self.ingc = GraphConvolutionBS(nfeat, nhid, activation, withbn, withloop)
            baseblockinput = nhid
        elif inputlayer == "none":
            self.ingc = lambda x: x
            baseblockinput = nfeat
        else:
            self.ingc = Dense(nfeat, nhid, activation)
            baseblockinput = nhid

        outactivation = lambda x: x
        if outputlayer == "gcn":
            self.outgc = GraphConvolutionBS(baseblockinput, nclass, outactivation, withbn, withloop)
        # elif outputlayer ==  "none": #here can not be none
        #    self.outgc = lambda x: x 
        else:
            self.outgc = Dense(nhid, nclass, activation)

        # hidden layer
        self.midlayer = nn.ModuleList()
        # Dense is not supported now.
        # for i in xrange(nhidlayer):
        for i in range(nhidlayer):
            gcb = self.BASEBLOCK(in_features=baseblockinput,
                                 out_features=nhid,
                                 nbaselayer=nbaselayer,
                                 withbn=withbn,
                                 withloop=withloop,
                                 activation=activation,
                                 dropout=dropout,
                                 dense=False,
                                 aggrmethod=aggrmethod)
            self.midlayer.append(gcb)
            baseblockinput = gcb.get_outdim()
        # output gc
        outactivation = lambda x: x  # we donot need nonlinear activation here.
        self.outgc = GraphConvolutionBS(baseblockinput, nclass, outactivation, withbn, withloop)

        self.reset_parameters()
        if mixmode:
            self.midlayer = self.midlayer.to(device)
            self.outgc = self.outgc.to(device)

    def reset_parameters(self):
        pass

    def forward(self, fea, adj):
        # input
        if self.mixmode:
            x = self.ingc(fea, adj.cpu())
        else:
            x = self.ingc(fea, adj)

        x = F.dropout(x, self.dropout, training=self.training)
        if self.mixmode:
            x = x.to(device)

        # mid block connections
        # for i in xrange(len(self.midlayer)):
        for i in range(len(self.midlayer)):
            midgc = self.midlayer[i]
            x = midgc(x, adj)
        # output, no relu and dropput here.
        x = self.outgc(x, adj)
        x = F.log_softmax(x, dim=1)
        return x


# Modified GCN
class GCNFlatRes(nn.Module):
    """
    (Legacy)
    """
    def __init__(self, nfeat, nhid, nclass, withbn, nreslayer, dropout, mixmode=False):
        super(GCNFlatRes, self).__init__()

        self.nreslayer = nreslayer
        self.dropout = dropout
        self.ingc = GraphConvolution(nfeat, nhid, F.relu)
        self.reslayer = GCFlatResBlock(nhid, nclass, nhid, nreslayer, dropout)
        self.reset_parameters()

    def reset_parameters(self):
        # stdv = 1. / math.sqrt(self.attention.size(1))
        # self.attention.data.uniform_(-stdv, stdv)
        # print(self.attention)
        pass

    def forward(self, input, adj):
        x = self.ingc(input, adj)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.reslayer(x, adj)
        # x = F.dropout(x, self.dropout, training=self.training)
        return F.log_softmax(x, dim=1)


