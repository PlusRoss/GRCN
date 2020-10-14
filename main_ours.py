import argparse
import glob
import logging
import sys
import os
import torch
from torch_scatter import scatter_add
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv
from complete import Complete, convert_edge2adj, normalize, _complete_acc
from utils import normalize_features, create_exp_dir, save_checkpoint, load_dataset
from models import GRCN, GRCN_fast
import importlib
import math
import time
import dataprocess
from torch.optim.lr_scheduler import *

import numpy as np
import random

# SYS_SEED = 42
EOS = 1e-10


def test_model(model, data, complete_model, args):
    model.eval()
    correct = 0
    for batch_data in dataprocess.dataloader(data, complete_model.loop_adj_part, args.batch_size, args.sparse):
        batch_data.to(device)
        _, pred = model(batch_data.x, batch_data.adj).max(dim=1)
        correct += pred[batch_data.test_mask].eq(batch_data.y[batch_data.test_mask]).sum().item()
    acc = correct / data.test_mask.sum().item()
    model.train()
    return acc

def eval_model(model, data, complete_model, args):
    model.eval()
    correct = 0
    loss = 0
    for batch_data in dataprocess.dataloader(data, complete_model.loop_adj_part, args.batch_size, args.sparse):
        batch_data.to(device)
        output = model(batch_data.x, batch_data.adj)
        _, pred = output.max(dim=1)
        correct += pred[batch_data.val_mask].eq(batch_data.y[batch_data.val_mask]).sum().item()
        loss += F.nll_loss(output[batch_data.val_mask], batch_data.y[batch_data.val_mask]).item()
    acc = correct / data.val_mask.sum().item()
    model.train()
    return acc, loss

def lr_decay(optimizer):
    pass


parser = argparse.ArgumentParser(description='PyTorch Enhance NC by GC Model')
parser.add_argument('--dataset', type=str, default='Cora',
                    help='dataset to use, [Cora, CiteSeer, PubMed]')
parser.add_argument("--sample", type=float, default=1.0,
                    help="sample ratio of edges")
parser.add_argument('--complete', type=str, default='None',
                    help='method for graph completion, [None, Graph, Both]')
parser.add_argument('--num', type=int, default=100,
                    help='number of edges to complete')
parser.add_argument('--seed', type=int, default=42,
                    help='ramdom seed for sampling')
parser.add_argument('--dataseed', type=int, default=42,
                    help='ramdom seed for data split')
parser.add_argument('--compl_param', type=str, default="no",
                    help='hyper parameters for completion method [10:1:2] use : to split')
parser.add_argument('--keep_train_num', action='store_true',
                    help='whether to fix the training label')
parser.add_argument('--save', action='store_true',
                    help='whether to save result, default False')
parser.add_argument('--dense', action='store_true',
                    help='whether to use dense adjacency matrix, default False')
parser.add_argument('--sparse', action='store_true',
                    help='whether to use sparse adjacency matrix, default False')
parser.add_argument('--reduce', type=str, default="knn",
                    help='method to reduce adj matrix, knn, threshold or topk')
parser.add_argument('--graphloss', action='store_true',
                    help='whether to use graph based unsupervised loss, default False')
parser.add_argument('--wd_graph', type=float, default=0.,
                    help='weight decay for graph learning parameters')
parser.add_argument('--alpha', type=float, default=1.,
                    help='weight of unsupervised loss')
parser.add_argument('--hid_graph', type=str, default="100:10",
                    help='hidden dimension of graph learning conv layer')
parser.add_argument('--patience', type=int, default=5,
                    help='patience for early stopping')
parser.add_argument('--train_num', type=int, default=20,
                    help='number of training labels per class')
parser.add_argument('--run_times', type=int, default=10,
                    help='Independent run times')

args = parser.parse_args()

config_file = "config.config_%s" % args.dataset
if args.dataseed == -1:
    params = importlib.import_module(config_file).params_fixed
else:
    params = importlib.import_module(config_file).params_random
args = argparse.Namespace(**vars(args), **params)

torch.random.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

if args.save:
    if args.dataset in ["CoraFull", "Computers", "Photo", "CS"]:
        nsave = "log/{}-{}/sample-{}/{}".format(args.dataset, args.train_num, args.sample, args.complete)
    else:
        if not args.keep_train_num:
            nsave = "log/{}/sample-{}/{}".format(args.dataset, args.sample, args.complete)
        else:
            nsave = "log/{}-keep/sample-{}/{}".format(args.dataset, args.sample, args.complete)
else:
    print("not saving file")
    nsave = "log/trash/{}".format(args.complete)
create_exp_dir(nsave)#, scripts_to_save=glob.glob('*.py'))
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p', filemode="w")
nfile = "para{}-nhid{}-lr{}-lrg{}-hidg{}-wd{}-dr{}-layer{}-norm{}-seed{}-{}".format(
        args.compl_param, args.nhid, args.lr, args.lr_graph, args.hid_graph, args.wd,
        args.dropout, args.layertype, args.normalize, args.seed, args.dataseed)
fh = logging.FileHandler(os.path.join(nsave, nfile + ".txt"), "w")
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

dataset = load_dataset(args.dataset)

logging.info('Original {} #nodes {} #edges {} #features {} #classes {}'.format
                (dataset, dataset[0].num_nodes, int(dataset[0].num_edges/2), dataset.num_features, dataset.num_classes))
ori_edge_index = dataset[0].edge_index

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device("cpu")
logging.info('Using {} for neural network training'.format(device))

sample_ratio = args.sample
sample_seed = args.seed
compl_param = args.compl_param.split(":")


compl_acc_list = []
deg_sum_all, deg_correct_all = 0, 0


sampled_edge_index = dataprocess.sample_edge(ori_edge_index, ratio=sample_ratio, seed=sample_seed)
logging.info('Sub-sampled {} #edges {}'.format(dataset, int(sampled_edge_index.shape[1]/2)))

data = dataset[0]
# data.name = args.dataset
dataprocess.random_split(data, args.dataseed, args)

model = eval(args.complete).Model(data.num_nodes, dataset.num_features, dataset.num_classes, device, args).to(device)

optimizer_base = torch.optim.Adam(model.base_parameters(), lr=args.lr, weight_decay=args.wd)
optimizer_graph = torch.optim.Adam(model.graph_parameters(), lr=args.lr_graph, weight_decay=args.wd_graph)

complete_device = torch.device("cpu")

complete_model = Complete(sampled_edge_index, data, model, complete_device, args)
compl_acc = 0.

model.train()
val_acc_epoch_list, best_val_acc = [], 0.
val_loss_epoch_list, best_val_loss = [], 1e10
final_test_acc = 0.
lr_base, lr_graph = args.lr, args.lr_graph


for epoch in range(args.epochs):
    # Optimize GCN
    train_loss = []
    for batch_data in dataprocess.dataloader(data, complete_model.loop_adj_part, args.batch_size, args.sparse):
        batch_data.to(device)
        optimizer_graph.zero_grad()
        optimizer_base.zero_grad()
        out = model(batch_data.x, batch_data.adj)
        loss = F.nll_loss(out[batch_data.train_mask], batch_data.y[batch_data.train_mask])
        train_loss.append(loss.item())
        loss.backward(retain_graph=False)
        if epoch % 1 == 0:
            optimizer_base.step()
        if epoch % 1 == 0:
            optimizer_graph.step()

    if (epoch+1) % args.log_epoch == 0:
        val_acc, val_loss = eval_model(model, data, complete_model, args)
        val_acc_epoch_list.append(val_acc)
        val_loss_epoch_list.append(val_loss)
        test_acc = test_model(model, data, complete_model, args)
        logging.info('Epoch: {} Loss: {:.3f} Val Acc: {:.3f} {:.3f} Test Vcc: {:.3f}'.format(
                    epoch, np.mean(train_loss), val_acc_epoch_list[-1], val_loss, test_acc))
        if val_acc_epoch_list[-1] > best_val_acc:
            best_val_acc = val_acc_epoch_list[-1]
            final_test_acc = test_acc
            logging.info("Update best val acc {:.3f} test vcc: {:.3f}".format(
                    best_val_acc, test_acc))

logging.info('Classification Val Acc: {:.2f}%'.format(best_val_acc*100))
logging.info('Classification Test Acc: {:.2f}%'.format(final_test_acc*100))
