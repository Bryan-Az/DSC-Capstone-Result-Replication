#!/usr/bin/env python
import sys
import os
import json
import yaml

sys.path.append('../src')

from src.q1_etl import get_data

from src.q1_model import DSNN, FCNN


def main(targets):
    '''
    Runs the main project pipeline logic, given the targets.
    targets available: 'test'. 
    
    `main` runs the target 'test'.
    '''
    #files for test and train in ~/teams
    with open('./config/data-params.json') as fh:
        data_cfg = json.load(fh)

    # make the data target
    train_generator, vtrain_generator, test_generator = get_data(**data_cfg)
    if 'test' in targets:
        output_dsnn = DSNN(train_generator, vtrain_generator, test_generator)
        output_fcnn = FCNN(train_generator, vtrain_generator, test_generator)
            
    output_dsnn.to_csv('./dsnn_predictions.csv', index = False)
    output_fcnn.to_csv('./fcnn_predictions.csv', index = False)
    
    print("Predictions written from pd.DataFrame to dsnn_predictions.csv")
    print("Predictions written from pd.DataFrame to fcnn_predictions.csv")

    return


if __name__ == '__main__':
    # run via:
    # python main.py data features model
    targets = sys.argv[1:]
    main(targets)
