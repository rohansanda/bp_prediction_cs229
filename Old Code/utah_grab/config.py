import torch
from torch import nn
import torch.optim as optim

# Random seed
SEED = 1

MODEL_TYPE = 'ResNet'

# Training params
TRAIN_PARAMS_ANN = {
    'tune_hyperparameters': False,
    'num_epochs': 50,
    'lr': [0.0005],
    'optimizers': [optim.SGD],
    'train_batch_size': 64,
    'val_batch_size': 64,
    'hidden_sizes':  [100],
    'num_layers': [2],
    'loss_function': nn.MSELoss(),
    'activation_functions': [nn.Tanh()]
}

#NeuralNetworkMod
# TRAIN_PARAMS_NN_MOD = {
#     'tune_hyperparameters': False,
#     'num_epochs': 75,
#     'lr': [5e-4],
#     'reg_lambda': [1e-4],
#     'optimizers': [optim.SGD],
#     'train_batch_size': 64,
#     'val_batch_size': 64,
# }

TRAIN_PARAMS_NN_MOD = {
    'tune_hyperparameters': False,
    'num_epochs': 50,
    'lr': [5e-4],
    'reg_lambda': [1e-3],
    'optimizers': [optim.Adam],
    'train_batch_size': 64,
    'val_batch_size': 64,
}

TRAIN_PARAMS_CNN = {
    'tune_hyperparameters': False,
    'num_epochs': 1,
    'lr': [1e-3],
    'optimizers': [optim.Adam],
    'train_batch_size': 64,
    'val_batch_size': 64,
    'loss_function': nn.MSELoss(),
    'activation_functions': [nn.Tanh()]
}

#cnn_1d_mod
# TRAIN_PARAMS_CNN_MOD = {
#     'tune_hyperparameters': True,
#     'num_epochs': 30,
#     'lr': [5e-4],
#     'reg_lambda': [1e-4, 5e-5, 1e-5, 5e-6, 1e-6],
#     'optimizers': [optim.Adam],
#     'train_batch_size': 64,
#     'val_batch_size': 64,
#     'loss_function': nn.MSELoss(),
# }

TRAIN_PARAMS_CNN_MOD = {
    'tune_hyperparameters': False,
    'num_epochs': 60,
    'lr': [5e-4],
    'reg_lambda': [5e-3],
    'optimizers': [optim.Adam],
    'train_batch_size': 64,
    'val_batch_size': 64,
    'loss_function': nn.MSELoss(),
}

#ResNet
TRAIN_PARAMS_RESNET = {
    'tune_hyperparameters': True,
    'num_epochs': 10,
    'lr': [1e-3, 1e-4, 5e-5], 
    'reg_lambda': [1e-2, 5e-3, 1e-4],
    'optimizers': [optim.Adam],
    'train_batch_size': 64,
    'val_batch_size': 64,
    'loss_function': nn.MSELoss()
}

# File locations for data
DATA_URI = '../data/'

DATA_FNAME = DATA_URI + 'test3_flat_segments.pickle'
LABELS_FNAME = DATA_URI + 'test3_flat_bps.pickle'

DATA_FNAMES = {
    'train_arguments': DATA_FNAME,
    'train_labels': LABELS_FNAME,
}

# Model saving
MODEL_URI = ''
MODEL_NAME = 'cnn_flat_lr2_test.pt'

SAVE_OPTIONS = {
    'save': True,
    'path': MODEL_URI + MODEL_NAME
}