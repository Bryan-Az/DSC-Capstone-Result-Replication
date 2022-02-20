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

from src.etl import get_data, read_test, load_graph 
import numpy as np

import pandas as pd

# ## Model is loaded from pretrained .pth file

# +
fileDir = os.path.dirname(os.path.realpath('__GENConv_model_best.pth__'))

filename = os.path.join(fileDir, './notebooks/GENConv_model_best.pth')

filename = os.path.abspath(filename)


# -

def main(targets):
    '''
    Runs the main project pipeline logic, given the targets.
    targets available: 'test'. 
    
    `main` runs the target 'test'.
    '''
    test_dataset = load_graph(train=False)
    batch_size = 32
    test_samples = len(test_dataset)
    #files for test and train in ~/teams
    with open('./config/data-params.json') as fh:
        data_cfg = json.load(fh)

    # make the data target (all)
    #train_generator, vtrain_generator, test_generator = get_data(**data_cfg)
    # make the data target (testing)
    test_loader = read_test(test_dataset)
    if 'test' in targets:
        model = GENConv_Classifier()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint)
        model.eval()
        t = tqdm(enumerate(test_loader),total=test_samples/batch_size)
        y_test = []
        y_predict = []
        for i,data in t:
            data = data.to(device)    
            batch_output = model(data.x, data.edge_index, data.batch)    
            y_predict.append(batch_output.detach().cpu().numpy())
            y_test.append(data.y.cpu().numpy())
        y_test = np.concatenate(y_test)
        y_predict = np.concatenate(y_predict)
        output = pd.Series(y_predict[:,0])
        output.to_csv('GENConv_predictions.csv', index = False)
        
    
    print("Predictions written from pd.Series to GENConv_predictions.csv")

    return

if __name__ == '__main__':
    # run via:
    # python main.py data features model
    targets = sys.argv[1:]
    main(targets)
