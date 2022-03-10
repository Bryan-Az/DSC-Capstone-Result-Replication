#!/usr/bin/env python
import sys
import os
import json
import yaml
from tqdm.notebook import tqdm

sys.path.append('../src')

import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from src.model import GENConv_Classifier 

sys.path.append(os.path.realpath('./config'))

from src.etl import get_data, read_test, load_graph 
import numpy as np

import pandas as pd

#imports
import torch
import torch.nn as nn
import torch_geometric
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from tqdm.notebook import tqdm
import numpy as np
import os.path as osp

# +
from torch_geometric.nn import GENConv

from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d
from torch_scatter import scatter_mean
# -

import tensorflow.keras as keras
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import uproot
from config.utils import *
import tensorflow
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, GlobalAveragePooling1D
import tensorflow.keras.backend as K
from tqdm.notebook import tqdm
import XRootD

from config.GraphDataset import GraphDataset

# +
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
# -

# ## Model is loaded from pretrained .pth file

# +
fileDir = os.path.dirname(os.path.realpath('__GENConv_model_best.pth__'))

filename = os.path.join(fileDir, './notebooks/GENConv_model_best.pth')

filename = os.path.abspath(filename)


# -

@torch.no_grad()
def test(model, loader, total, batch_size, leave=False):
    model.eval()
    
    #xentropy = nn.CrossEntropyLoss(reduction='mean')
    xentropy = nn.BCEWithLogitsLoss()

    sum_loss = 0.
    #tqdm creates progress bar based on the batch
    t = tqdm(enumerate(loader), total=total/batch_size, leave=leave)
    for i, data in t:
        data = data.to(device)
        batch_output = model(data.x, data.edge_index, data.batch)
        batch_output = batch_output.float()
        batch_loss_item = xentropy(batch_output, data.y.float()).item()
        sum_loss += batch_loss_item
        t.set_description("loss = %.5f" % (batch_loss_item))
        t.refresh() # to show immediately the update

    return sum_loss/(i+1)


def train(model, optimizer, loader, total, batch_size, leave=False):
    model.train()

    #xentropy = nn.CrossEntropyLoss(reduction='mean')
    xentropy = nn.BCEWithLogitsLoss()

    sum_loss = 0.
    t = tqdm(enumerate(loader), total=total/batch_size, leave=leave)
    for i, data in t:
        data = data.to(device)
        optimizer.zero_grad()
        batch_output = model(data.x, data.edge_index, data.batch)
        batch_output = batch_output.float()
        batch_loss = xentropy(batch_output, data.y.float())
        batch_loss.backward()
        batch_loss_item = batch_loss.item()
        t.set_description("loss = %.5f" % batch_loss_item)
        t.refresh() # to show immediately the update
        sum_loss += batch_loss_item
        optimizer.step()

    return sum_loss/(i+1)


def main(targets):
    '''
    Runs the main project pipeline logic, given the targets.
    targets available: 'test'. 
    
    `main` runs the target 'test'.
    '''
    batch_size = 32
    #files for test and train in ~/teams
    with open('./config/data-params.json') as fh:
        data_cfg = json.load(fh)

    # make the data target (all)
    train_generator, vtrain_generator, test_generator = get_data(**data_cfg)
    test_samples = test_generator.dataset.len()
    train_samples = len(train_generator.dataset)
    
    model = GENConv_Classifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()
    if 'train' in targets:
        
        n_epochs = 10
        stale_epochs = 0
        best_valid_loss = 99999
        patience = 5
        t = tqdm(range(0, n_epochs))

        for epoch in t:
            loss = train(model, optimizer, train_generator, train_samples, batch_size, leave=bool(epoch==n_epochs-1))
            valid_loss = test(model, valid_loader, valid_samples, batch_size, leave=bool(epoch==n_epochs-1))
            print('Epoch: {:02d}, Training Loss:   {:.4f}'.format(epoch, loss))
            print('           Validation Loss: {:.4f}'.format(valid_loss))

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                modpath = osp.join('GENConv_model_best.pth')
                print('New best model saved to:',modpath)
                torch.save(model.state_dict(),modpath)
                stale_epochs = 0
            else:
                print('Stale epoch')
                stale_epochs += 1
            if stale_epochs >= patience:
                print('Early stopping after %i stale epochs'%patience)
                break          
        
    if 'test' in targets:
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint)
        model.eval()
        t = tqdm(enumerate(test_generator),total=test_samples/batch_size)
        y_test = []
        y_predict = []
        spec_gnn = []
        for i,data in t:
            data = data.to(device)    
            batch_output = model(data.x, data.edge_index, data.batch)
            spec_gnn.append(data.u)
            y_predict.append(batch_output.detach().cpu().numpy())
            y_test.append(data.y.cpu().numpy())
        y_test = np.concatenate(y_test)
        y_predict = np.concatenate(y_predict)
        spec_gnn = np.concatenate(spec_gnn, axis=0)
        output = pd.DataFrame({'qcd_prediction':y_predict[:,0],'hbb_prediction': y_predict[:,1],'truth_hbb_label': y_test[:,1], 'truth_qcd_label': y_test[:, 0], 'Mass':spec_gnn[:,0], 'Momentum_pt': spec_gnn[:,1]})
        output.to_csv('GENConv_predictions.csv', index = False)
        
    
    print("Predictions written from pd.Series to GENConv_predictions.csv")

    return

if __name__ == '__main__':
    # run via:
    # python main.py data features model
    targets = sys.argv[1:]
    main(targets)
