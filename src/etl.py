import os
import sys
#data to be tested on deep sets classifier

train_dir = '~/teams/DSC180A_FA21_A00/a11/train/'
test_dir = '~/teams/DSC180A_FA21_A00/a11/test/'

sys.path.append(os.getcwd() + '/config')
from config.utils import get_file_handler 
from DataGenerator import DataGenerator
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

#returns tuple of (train, test) data
def get_data(train_file, vtrain_file, test_file):
    '''
    create generators for train/test 
    '''
    train_fp = os.path.join(train_dir, train_file)
    vtrain_fp = os.path.join(train_dir, vtrain_file)
    test_fp = os.path.join(test_dir, test_file)

    return (read_train(train_fp), read_train(vtrain_fp), read_test(test_fp))

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


#returns test generator
def read_test(fp):
    '''
    Reads raw test data from disk.
    (Would normally be more complicated!)
    '''
    file_set = [fp]
    
    test_generator = DataGenerator(file_set, features, labels, spectators, batch_size=1024, n_dim=ntracks, 
                               remove_mass_pt_window=True, 
                               remove_unlabeled=True)
    return test_generator