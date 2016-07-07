from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from input_output import load_data
import numpy as np
from keras.utils import np_utils

DATAFILE = "SEQC_NB_batchCorr_tr_FAV_TopFeats.txt"
LABELSFILE = "SEQC_NB_batchCorr_tr_FAV.lab"
sample_names_tr, var_names_tr, X_train = load_data(DATAFILE)
y_tr = np.loadtxt(LABELSFILE, dtype=np.int)
n_classes = np.max(y_tr)+1
y_train = np_utils.to_categorical(y_tr+1)
y_train = np_utils.to_categorical(y_tr,n_classes)

model = Sequential()
model.add(Dense(64,input_dim=X_train.shape[1],init='uniform'))
model.add(Activation('tanh'))
model.add(Dense(64,init='uniform'))
model.add(Activation('tanh'))
model.add(Dense(2,init='uniform'))
model.add(Activation('softmax'))

sgd = SGD(lr=0.003)
model.compile(loss='binary_crossentropy',optimizer=sgd,metrics=['accuracy'])
model.fit(X_train,y_train,nb_epoch=200,batch_size=16)
