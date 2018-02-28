import os
import csv
import numpy as np
from keras.callbacks import ModelCheckpoint
#from utils import to_categorical, get_comment_ids
import nltk_utils
from vdcnn import build_model
import pandas as pd
from model.models import vdcnn_model

def get_input_data(file_path):
    tr = pd.read_csv(file_path)
    
    X_train = np.zeros((tr.shape[0], 512))
    
    for i, comment in enumerate(tr.comments.values):
        X_train[i] = nltk_utils.get_comment_ids(comment)
    
    y_train = tr['class'].values
    
    return X_train, y_train


def train(input_file, max_feature_length, num_classes, batch_size, num_epochs, save_dir=None, print_summary=False):
    # Stage 1: Convert raw texts into char-ids format && convert labels into one-hot vectors
    X_train, y_train = get_input_data(input_file)
    y_train = nltk_utils.to_categorical(y_train, num_classes)

    # Stage 2: Build Model


    #model = build_model(num_filters=num_filters, num_classes=num_classes)

    model = vdcnn_model()

    # Stage 3: Training
    save_dir = save_dir if save_dir is not None else 'checkpoints'
    filepath = os.path.join(save_dir, "weights-{epoch:02d}-{val_acc:.2f}.hdf5")
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    if print_summary:
        print(model.summary())

    model.fit(
        x=X_train,
        y=y_train,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_split=0.33,
        callbacks=[checkpoint],
        shuffle=True,
        verbose=True
    )
train = train('data/train.csv', nltk_utils.FEATURE_LEN, num_classes=10, batch_size=16, num_epochs=3)
    
#tr = get_input_data('data/train.csv')