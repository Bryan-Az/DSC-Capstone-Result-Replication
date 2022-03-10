import os
import sys
#data to be tested on deep sets classifier

#adding config path for importing GraphDataset class
sys.path.insert(0, "../config")

sys.path.append('/home/baambriz/q2_code/')
from config.utils import get_file_handler 
from config.DataGenerator import DataGenerator
import torch
import yaml

from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataListLoader
from torch.utils.data import random_split

from config.GraphDataset import GraphDataset

# +
fileDir = os.path.dirname(os.path.realpath('__definitions.yml__'))

filename = os.path.join(fileDir, './config/definitions.yml')

filename = os.path.abspath(filename)


# -

def collate(items):
        l = sum(items, [])
        return Batch.from_data_list(l)


with open(filename) as file:
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

train_dir = '~/teams/DSC180A_FA21_A00/a11/train/'
test_dir = '~/teams/DSC180A_FA21_A00/a11/test/'

batch_size = 32


# ## Create and load graph dataset

def load_graph(filenames, train = True):
    if train:
        graph_dataset = GraphDataset('gdata_train', features, labels, spectators, n_events=15000, n_events_merge=1, 
                                     file_names=filenames)
        return graph_dataset
    else:
        test_dataset = GraphDataset('gdata_test', features, labels, spectators, n_events=15000, n_events_merge=1, 
                                     file_names=filenames)
        return test_dataset


# ## Testing is done using the pretrained model

#returns test generator
def read_test(fp):
    '''
    Reads raw test data from disk.
    (Would normally be more complicated!)
    '''
    test_dataset = load_graph(filenames=fp,train=False)
    test_loader = DataListLoader(test_dataset, batch_size=batch_size, pin_memory=True, shuffle=False)
    test_loader.collate_fn = collate
    return test_loader


# ## Training writes best model weights for use in testing

#returns train generator
def read_train(fp):
    '''
    Reads raw training data from disk.
    (Would normally be more complicated!)
    '''
    graph_dataset = load_graph(filenames=fp, train=True)
    torch.manual_seed(0)
    valid_frac = 0.20
    full_length = len(graph_dataset)
    valid_num = int(valid_frac*full_length)
    
    train_dataset, valid_dataset = random_split(graph_dataset, [full_length-valid_num,valid_num])
    train_loader = DataListLoader(train_dataset, batch_size=batch_size, pin_memory=True, shuffle=True)
    train_loader.collate_fn = collate
    valid_loader = DataListLoader(valid_dataset, batch_size=batch_size, pin_memory=True, shuffle=False)
    valid_loader.collate_fn = collate
    return train_loader, valid_loader


# +

#returns tuple of (train, test) data
def get_data(train_file, test_file):
    '''
    create generators for train/test 
    '''
    train_fp = [os.path.join(train_dir, train_file)]
    test_fp = [os.path.join(test_dir, test_file)]
    train_loader, valid_loader = read_train(train_fp)
    test_loader = read_test(test_fp)
    return (train_loader, valid_loader, test_loader)
# -


