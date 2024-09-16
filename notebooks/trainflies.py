### Import packages ########################################################################################

# Run with the newest versions of all codes without restart (typically used in ipynb)
# %load_ext autoreload
# %autoreload 2
import os
import sys
sys.path.append('../')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "3, 4"
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
from scipy.io import loadmat
import h5py
from copy import copy
import random
import pickle

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

### Import Modules ############################################################################################

from rnn_toolkit.models import LSTMModel
from rnn_toolkit.data import preprocessing, batch_data
from rnn_toolkit.evaluation import persisenceplot, transitionplot, eigenvalueplot, stability_checker, loss_plot
from rnn_toolkit.training import evaluate, loss, optimizer, train_model, train

### Set up Parameters ###############################################################################################

name_mat='../../../behaviorTimeSeries.mat'
states='reduced_states'

numFlies=59
numstates=117
sequence_length=500
n_neurons=512
batch_size=59
lag=200

### Data Preprocessing ###################################################################################################

dt = preprocessing(name_mat, states, numFlies, numstates, minlength = 500000)
input_train, input_test, target_train, target_test = batch_data(dt, batch_size, sequence_length, numstates, lag, v = 0.8, one_hot_target = False)

train_dataset = TensorDataset(input_train, target_train)
test_dataset = TensorDataset(input_test, target_test)

train_loader = DataLoader(train_dataset, batch_size=59, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=59, shuffle=False)

### Load LSTM Model ###########################################################################################################

hidden_sizes = [128, 128, 128]
model = LSTMModel(numstates, n_neurons, batch_size, hidden_sizes, dropout = 0.1)

### Train Model ##############################################################################################################
optimizer = optimizer(model, lr = 0.05)
criterion = nn.CrossEntropyLoss()  
model, hidden_states_h, hidden_states_c, train_loss, test_loss = train_model(model, train_loader, test_loader, optimizer, batch_size, sequence_length, criterion, gradient_clipping = True, normalization = True, num_epochs=1000)

### Evaluate Model ###############################################################################################################

# Loss Pot
loss_plot(train_loss, test_loss, time_steps=1000)

# Hidden States plot
stability_checker(hidden_states_h, hidden_states_c, time_steps=1000)

# Generate Sequences

# Plots



