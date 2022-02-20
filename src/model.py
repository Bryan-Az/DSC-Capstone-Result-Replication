# -*- coding: utf-8 -*-
import sys

import tensorflow.keras as keras
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import uproot
import tensorflow
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, GlobalAveragePooling1D
import tensorflow.keras.backend as K
from tqdm.notebook import tqdm
import XRootD
from tensorflow.keras.layers import Flatten
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import pandas as pd
import os
import yaml

with open('./config/definitions.yml') as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
    definitions = yaml.load(file, Loader=yaml.FullLoader)

features = definitions['features']
spectators = definitions['spectators']
labels = definitions['labels']

nfeatures = definitions['nfeatures']
nspectators = definitions['nspectators']
nlabels = definitions['nlabels']
ntracks = definitions['ntracks']

import torch
import torch.nn as nn
import torch_geometric
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from tqdm.notebook import tqdm
import numpy as np

# +
inputs = 48
hidden = 128
outputs = 1

from torch_geometric.nn import GENConv
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d
from torch_scatter import scatter_mean

class GENConv_Classifier(nn.Module):

    def __init__(self, width = hidden, n_inputs = inputs):
        super(GENConv_Classifier, self).__init__()
        self.width = width
        self.act = nn.ReLU

        # Initial linear layers
        self.nn1 = nn.Sequential(
            self.act(),
            nn.Linear(n_inputs, width),
            self.act(),
            nn.Linear(width, width),
            self.act(),
            nn.Linear(width, width)                   
        )
        # Generalized Convolutional layer
        self.conv = GENConv(width, width, num_layers=2, t=1, learn_t=True)

        # Pre-final linear layers
        self.nn2 = nn.Sequential(
            nn.Linear(n_inputs, width),
            self.act(),
            nn.Linear(width, width),
            self.act(),
            nn.Linear(width, width),
            self.act(),
            nn.Linear(width, width),
        )
        self.bn = BatchNorm1d(n_inputs)
        # output layer
        self.output = nn.Linear(width, outputs)

    def forward(self, X, edge_index, batch):
        #Normalization → ReLU → GraphConv → Addition
        x = self.bn(X)
        # input layer
        x1 = self.nn1(x)
        #GENConv
        x2 = self.conv(x1, edge_index)
        x3 = scatter_mean(x, batch, dim=0)
        # hidden layers
        x4 = self.nn2(x3)

        # output layer
        x = self.output(x4)
        return x
