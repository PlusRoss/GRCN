import torch
import torch.nn.functional as F

params_random={
    "nhid": 64, # number of hidden units per layer
    "dropout": 0.5,
    "F": torch.relu,
    "F_graph": "identity",
    "lr": 5e-3, # learning rate for node classification
    "wd": 5e-3, # weight decay for node classification
    "lr_graph": 5e-3, # learning rate for graph modification
    "epochs": 300, # epoch number for model training
    "log_epoch": 10,
    "normalize": True,
    "batch_size": 20000,
    "layertype": "diag"
}
