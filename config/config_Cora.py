import torch
import torch.nn.functional as F

params_fixed={
    "nhid": 32, # number of hidden units per layer
    "dropout": 0.5,
    "F": torch.relu,
    "F_graph": torch.tanh,
    "lr": 5e-3, # learning rate for node classification
    "wd": 5e-3, # weight decay for node classification
    "lr_graph": 1e-3, # learning rate for graph modification
    "epochs": 300, # epoch number for model training
    "log_epoch": 1,
    "normalize": True,
    "batch_size": 3000,
    "layertype": "diag"
}

params_random={
    "nhid": 32, # number of hidden units per layer
    "dropout": 0.5,
    "F": torch.relu,
    "F_graph": torch.tanh,
    "lr": 5e-3, # learning rate for node classification
    "wd": 5e-3, # weight decay for node classification
    "lr_graph": 1e-3, # learning rate for graph modification
    "epochs": 300, # epoch number for model training
    "log_epoch": 1,
    "normalize": True,
    "batch_size": 3000,
    "layertype": "diag"
}
