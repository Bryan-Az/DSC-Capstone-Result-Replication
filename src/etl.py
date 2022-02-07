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

filename

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

# ## Create and load graph dataset

def load_graph(local = True, train = True):
    # Load Dataset
    if local:
        file_names = ['~/teams/DSC180A_FA21_A00/a11/train/ntuple_merged_10.root']
        file_names_test = ['~/teams/DSC180A_FA21_A00/a11/test/ntuple_merged_0.root']
    else:
        file_names = ['root://eospublic.cern.ch//eos/opendata/cms/datascience/HiggsToBBNtupleProducerTool/HiggsToBBNTuple_HiggsToBB_QCD_RunII_13TeV_MC/train/ntuple_merged_10.root']
        file_names_test = ['root://eospublic.cern.ch//eos/opendata/cms/datascience/HiggsToBBNtupleProducerTool/HiggsToBBNTuple_HiggsToBB_QCD_RunII_13TeV_MC/test/ntuple_merged_0.root']
    
    if train:
        graph_dataset = GraphDataset('gdata_train', features, labels, spectators, n_events=1000, n_events_merge=1, 
                                     file_names=file_names)
        return graph_dataset
    else:
        test_dataset = GraphDataset('gdata_test', features, labels, spectators, n_events=2000, n_events_merge=1, 
                                     file_names=file_names_test)
        return test_dataset


# ## Generators are created all in one function (TODO)

#returns train/test/validation generator tuple
def create_generators(graph):
    
    def collate(items):
        l = sum(items, [])
        return Batch.from_data_list(l)

    torch.manual_seed(0)
    valid_frac = 0.20
    full_length = len(graph_dataset)
    valid_num = int(valid_frac*full_length)
    batch_size = 32

    train_dataset, valid_dataset = random_split(graph_dataset, [full_length-valid_num,valid_num])

    train_loader = DataListLoader(train_dataset, batch_size=batch_size, pin_memory=True, shuffle=True)
    train_loader.collate_fn = collate
    valid_loader = DataListLoader(valid_dataset, batch_size=batch_size, pin_memory=True, shuffle=False)
    valid_loader.collate_fn = collate
    test_loader = DataListLoader(test_dataset, batch_size=batch_size, pin_memory=True, shuffle=False)
    test_loader.collate_fn = collate
    
    return (train_loader, test_loader, valid_loader)


# ## Testing is done using the pretrained model

#returns test generator
def read_test(graph):
    '''
    Reads raw test data from disk.
    (Would normally be more complicated!)
    '''
    def collate(items):
        l = sum(items, [])
        return Batch.from_data_list(l)
    torch.manual_seed(0)
    valid_frac = 0.20
    full_length = len(graph)
    valid_num = int(valid_frac*full_length)
    batch_size = 32
    
    test_loader = DataListLoader(graph, batch_size=batch_size, pin_memory=True, shuffle=False)
    test_loader.collate_fn = collate
    return test_loader


read_test(load_graph(train=False))


# ## TODO: Enable the training and loading of data

# +

#returns tuple of (train, test) data
def get_data(train_file, vtrain_file, test_file):
    '''
    create generators for train/test 
    '''
    train_fp = os.path.join(train_dir, train_file)
    vtrain_fp = os.path.join(train_dir, vtrain_file)
    test_fp = os.path.join(test_dir, test_file)

    return (read_train(train_fp), read_train(vtrain_fp), read_test(test_fp))


# -

#returns train generator
def read_train(fp):
    '''
    Reads raw training data from disk.
    (Would normally be more complicated!)
    '''
    file_set = [fp]
    
    train_generator = DataGenerator(file_set, features, labels, spectators, batch_size=1024, n_dim=ntracks, 
                                remove_mass_pt_window=False, 
                                remove_unlabeled=True, max_entry=8000)

    return train_generator
