import torch
import torch.nn as nn
import os, shutil
import numpy as np
from torch_geometric.datasets import Planetoid, CoraFull, Amazon, Coauthor
import scipy.sparse as sp


def load_dataset(name):
    if name in ["Cora", "CiteSeer", "PubMed"]:
        dataset = Planetoid(root='./data/'+name, name=name)
    elif name == "CoraFull":
        dataset = CoraFull(root='./data/'+name)
    elif name in ["Computers", "Photo"]:
        dataset = Amazon(root='./data/'+name, name=name)
    elif name in ["CS", "Physics"]:
        dataset = Coauthor(root='./data/'+name, name=name)
    else:
        exit("wrong dataset")
    return dataset

def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = mx.sum(1)
    r_inv = torch.pow(rowsum, -1).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = torch.diag(r_inv)
    mx = torch.mm(r_mat_inv, mx)
    return mx


def create_exp_dir(path, scripts_to_save=None):
    path_split = path.split("/")
    path_i = "."
    for one_path in path_split:
        path_i += "/" + one_path
        if not os.path.exists(path_i):
            os.mkdir(path_i)

    print('Experiment dir : {}'.format(path_i))
    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


def save_checkpoint(model, optimizer, epoch, path, finetune=False):
    if finetune:
        torch.save(model, os.path.join(path, 'finetune_model.pt'))
        torch.save(optimizer.state_dict(), os.path.join(path, 'finetune_optimizer.pt'))
    else:
        torch.save(model, os.path.join(path, 'model.pt'))
        torch.save(optimizer.state_dict(), os.path.join(path, 'optimizer.pt'))
    torch.save({'epoch': epoch+1}, os.path.join(path, 'misc.pt'))
