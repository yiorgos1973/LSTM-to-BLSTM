import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow
import keras
import random

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score, log_loss
from sklearn.model_selection import train_test_split
from modules.text_preprocessor import TextsPreprocessor
from modules.vectorizer import Vectorizer
from modules.stopwords import Stopwords

from keras.models import Sequential
from keras.layers import Bidirectional
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Conv1D
from keras.layers.pooling import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.layers import Dropout

def create_model(embedding_kwargs={}, conv1d_kwargs={}, lstm_kwargs={}, 
                dropout_after_embedding=0.0,
                dropout_after_conv1d=0.0,
                dropout_after_lstm=0.0,
                compile_kws={},
                bidirectional=False):
    model = Sequential()
    model.add(Embedding(**embedding_kwargs))
    model.add(Dropout(dropout_after_embedding))
    model.add(Conv1D(**conv1d_kwargs))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(dropout_after_conv1d))
    lstm = LSTM(**lstm_kwargs)
    if bidirectional:
        lstm = Bidirectional(lstm)
    model.add(lstm)
    model.add(Dropout(dropout_after_lstm))
    model.add(Dense(1, activation='sigmoid'))
    cml = dict(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    for kw in compile_kws:
        cml[kw] = compile_kws[kw]
    model.compile(**cml)
    return model

def create_lstm(embedding_kwargs={}, conv1d_kwargs={}, lstm_kwargs={}, dropout_after_embedding=0.0, compile_kws={}):
    lstm = Sequential()
    lstm.add(Embedding(**embedding_kwargs))
    lstm.add(Dropout(dropout_after_embedding))
    lstm.add(Conv1D(**conv1d_kwargs))
    lstm.add(MaxPooling1D(pool_size=2))
    lstm.add(LSTM(**lstm_kwargs)) 
    lstm.add(Dense(1, activation='sigmoid'))
    cml = dict(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    for kw in compile_kws:
        cml[kw] = compile_kws[kw]
    lstm.compile(**cml)
    return lstm

def create_blstm(embedding_kwargs={}, conv1d_kwargs={}, lstm_kwargs={}, dropout_after_embedding=0.0, compile_kws={}):
    blstm = Sequential()
    blstm.add(Embedding(**embedding_kwargs))
    blstm.add(Dropout(dropout_after_embedding))
    blstm.add(Conv1D(**conv1d_kwargs))
    blstm.add(MaxPooling1D(pool_size=2))
    blstm.add(Bidirectional(LSTM(**lstm_kwargs,)))
    blstm.add(Dense(1, activation='sigmoid'))
    cml = dict(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    for kw in compile_kws:
        cml[kw] = compile_kws[kw]
    blstm.compile(**cml)
    return blstm

def evaluate(model, X_test, y_test):
    y_pred = model.predict_classes(X_test)
    proba = np.ravel(model.predict_proba(X_test))
    cnf = confusion_matrix(y_test, y_pred)     
    report = classification_report(y_test, y_pred)
    print("Confusion Matrix\n"
          "----------------\n{}".format(cnf), end='\n\n')
    print("Classification Report".center(70))
    print("---------------------".center(70)) 
    print(report)
    cnf_values = dict(zip(['tn', 'fp', 'fn', 'tp'], cnf.ravel()))
    logloss = log_loss(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    accuracy = accuracy_score(y_test, y_pred)
    return dict(cnf=cnf_values, logloss=logloss, auc=auc, acc=accuracy)
    

def plot_history(history, title=None, context="fivethirtyeight"):
    val_loss = history.history['val_loss']
    val_acc = history.history['val_accuracy']
    acc = history.history['accuracy']
    loss = history.history['loss']
    return plot_history_(loss, val_loss, acc, val_acc, contex)

def plot_history_(loss, val_loss, acc, val_acc, title='', context='fivethirtyeight'):
    n_epochs = np.arange(1, len(loss)+1)
    with plt.style.context(context):
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6), sharex=True)
        ax1.plot(n_epochs, loss, marker='o', markersize=10, lw=2, label='training')
        ax1.plot(n_epochs, val_loss, marker='*', lw=2, markersize=10, ls=':', label='validation')
        ax2.plot(n_epochs, acc, marker='o', markersize=10, lw=2, label='training')
        ax2.plot(n_epochs, val_acc, marker='*', lw=2, markersize=10, ls=':', label='validation')
        ax1.set_xlabel("Epoch")
        ax2.set_xlabel("Epoch")
        ax1.set_ylabel("Log Loss", labelpad=10)
        ax2.set_ylabel("Accuracy", labelpad=10)
        ax1.set_title("Log Loss History")
        ax2.set_title("Accuracy History")
        plt.xticks(n_epochs, n_epochs)
        f.legend(*ax1.get_legend_handles_labels(), fancybox=True, shadow=True, bbox_to_anchor=(0.57, 0.97))
        f.suptitle(title, y=1.05, fontsize='xx-large')
        f.tight_layout()
    return f