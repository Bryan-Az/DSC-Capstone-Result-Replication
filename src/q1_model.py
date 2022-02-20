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

# +
fileDir = os.path.dirname(os.path.realpath('__definitions.yml__'))

filename = os.path.join(fileDir, './config/definitions.yml')

filename = os.path.abspath(filename)
# -

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

#FC Neural network
def FCNN(train_generator, vtrain_generator, test_generator):
    # define dense keras model
    inputs = Input(shape=(ntracks, nfeatures,), name='input')  
    x = BatchNormalization(name='bn_1')(inputs)
    x = Flatten(name='flatten_1')(x)
    x = Dense(64, name='dense_1', activation='relu')(x)
    x = Dense(32, name='dense_2', activation='relu')(x)
    x = Dense(32, name='dense_3', activation='relu')(x)
    outputs = Dense(nlabels, name='output', activation='softmax')(x)
    keras_model_dense = Model(inputs=inputs, outputs=outputs)
    keras_model_dense.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    reduce_lr = ReduceLROnPlateau(patience=5, factor=0.5)
    model_checkpoint = ModelCheckpoint('keras_model_dense_best.h5', monitor='val_loss', save_best_only=True)
    callbacks = [early_stopping, model_checkpoint, reduce_lr]

    # fit keras model
    history_dense = keras_model_dense.fit(train_generator,
                                          validation_data=vtrain_generator,
                                          steps_per_epoch=len(train_generator),
                                          validation_steps=len(vtrain_generator),
                                          max_queue_size=5,
                                          epochs=20,
                                          shuffle=False,
                                          callbacks=callbacks,
                                          verbose=0)
    # reload best weights
    keras_model_dense.load_weights('keras_model_dense_best.h5')
    # run model inference on test data set

    predict_array_fcnn = []
    label_array_test = []

    for t in test_generator:
        label_array_test.append(t[1])
        predict_array_fcnn.append(keras_model_dense.predict(t[0]))
        

    predict_array_fcnn = np.concatenate(predict_array_fcnn, axis=0)
    label_array_test = np.concatenate(label_array_test, axis=0)

    preds = pd.DataFrame(predict_array_fcnn)
    truth_label = pd.DataFrame(label_array_test)
    output = pd.concat([preds, truth_label], 1)
    output.columns = ['hbb_prediction', 'qcd prediction', 'hbb_label', 'qcd_label']
    return output


#Deep set Neural network
def DSNN(train_generator, vtrain_generator, test_generator): 
    
    #DSNN part
    # define Deep Sets model with Dense Keras layer
    inputs = Input(shape=(ntracks, nfeatures,), name='input')  
    x = BatchNormalization(name='bn_1')(inputs)
    x = Dense(64, name='dense_1', activation='relu')(x)
    x = Dense(32, name='dense_2', activation='relu')(x)
    x = Dense(32, name='dense_3', activation='relu')(x)
    # sum over tracks
    x = GlobalAveragePooling1D(name='pool_1')(x)
    x = Dense(100, name='dense_4', activation='relu')(x)
    outputs = Dense(nlabels, name='output', activation='softmax')(x)
    keras_model_deepset = Model(inputs=inputs, outputs=outputs)
    keras_model_deepset.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    reduce_lr = ReduceLROnPlateau(patience=5, factor=0.5)
    model_checkpoint = ModelCheckpoint('keras_model_deepset_best.h5', monitor='val_loss', save_best_only=True)
    callbacks = [early_stopping, model_checkpoint, reduce_lr]

    # fit keras model
    history_deepset = keras_model_deepset.fit(train_generator, 
                                              validation_data=vtrain_generator, 
                                              steps_per_epoch=len(train_generator), 
                                              validation_steps=len(vtrain_generator),
                                              max_queue_size=5,
                                              epochs=20, 
                                              shuffle=False,
                                              callbacks=callbacks, 
                                              verbose=0)
    # reload best weights
    keras_model_deepset.load_weights('keras_model_deepset_best.h5')

    def find_nearest(array,value):
        idx = (np.abs(array-value)).argmin()
        return idx, array[idx]
    # run model inference on test data set

    predict_array_deepset = []
    label_array_test = []

    for t in test_generator:
        label_array_test.append(t[1])
        predict_array_deepset.append(keras_model_deepset.predict(t[0]))
        

    predict_array_deepset = np.concatenate(predict_array_deepset, axis=0)
    label_array_test = np.concatenate(label_array_test, axis=0)

    preds = pd.DataFrame(predict_array_deepset)
    truth_label = pd.DataFrame(label_array_test)
    output = pd.concat([preds, truth_label], 1)
    output.columns = ['hbb_prediction', 'qcd prediction', 'hbb_label', 'qcd_label']
    return output
