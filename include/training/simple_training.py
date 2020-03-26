# use 'model checkpoint' to retrace best prior model
import io
import os
import sys
import warnings

warnings.filterwarnings("ignore")
text_trap = io.StringIO()
printing_active = True

running_model = None
df_captions = None
df_image = None

print('\033[31m' + "WARNING: IN THIS FILE, ALL WARNINGS ARE DEACTIVATED" + '\033[0m')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def interface_print(string, keep_line=False, red=False):
    global printing_active
    if red:
        string = '\33[31m' + string + '\33[0m'
    if printing_active:
        print(string, end='', flush=True)
        sys.stdout = text_trap
        printing_active = False
    elif not keep_line:
        sys.stdout = sys.__stdout__
        print(string)
        printing_active = True
    else:
        sys.stdout = sys.__stdout__
        print(string, end='', flush=True)
        sys.stdout = text_trap


def single_line_print(string, red=False):
    global printing_active
    if red:
        string = '\33[31m' + string + '\33[0m'
    if printing_active:
        print(string)
    else:
        printing_active = True
        sys.stdout = sys.__stdout__
        print(string)


interface_print("performing imports... ")
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sparse
from scipy import spatial
# keras
from sklearn.model_selection import train_test_split
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
# own code
from include.networks import network as nw

interface_print("done")

"""---------------------------------concerning data to be worked with------------------------------------------------"""


def load_data(caption_file_path, image_features_path):
    """
    :param caption_file_path: directory of the file containing the caption features
    :param image_features_path: directory of the file containing the corresponding image features
    :return:
    """
    global df_captions, df_image
    try:
        interface_print('importing caption file... ')
        df_captions = sparse.load_npz(caption_file_path)
        interface_print('done')
        interface_print('importing image features... ')
        df_image = pd.read_csv(image_features_path, sep=" ", header=None)
        interface_print('done')
    except Exception as e:
        interface_print("DATA IMPORT FAILED", red=True)
        interface_print("underlining exception was: ")
        interface_print(str(e), red=True)


def default_data_load():
    """
    loads the data from the default directories
    :return:
    """
    caption_path = 'include/data/caption_features.npz'
    feature_path = "include/data/image_features.csv"
    load_data(caption_path, feature_path)


# %% subset captions and image to start with few examples
num_samples = 2000
X_captions_subset = df_captions[0:num_samples, :][::5].todense().astype(float)
y_image_subset = df_image.iloc[0:int(num_samples / 5), :].values

# make train and validation set (test set is for later once we have found good parameters)
val_size = round(num_samples / 5 * 0.25)
X_train, X_val, y_train, y_val = train_test_split(X_captions_subset,
                                                  y_image_subset, test_size=val_size)

print(f'Size train X: {X_train.shape}, train y labels {y_train.shape}')
print(f'Size validation X: {X_val.shape}, validation y labels {y_val.shape}')

# %% define model architecture (play with this)


"""---------------------------------concerning current model/the model used------------------------------------------"""


def import_model(file_path):
    """
    :param file_path: the directorie of the model to be loaded
    :return: no return value is given, if filepath ok, running_model will be altered
    """
    global running_model
    if file_path[-3:] != ".h5":
        single_line_print("ERROR: file of '.h5' format was expected, but "
                          + file_path[-3:] + " was given, NO NEW MODEL LOADED", red=True)
        return
    try:
        interface_print('importing model... ')
        running_model = nw.import_network(file_path)
        interface_print('done')
        in_size = running_model.layers[0].input_shape[1]
        out_size = running_model.layers[len(running_model.layers) - 1].output_shape[1]
        single_line_print("new model of form MODEL: R_" + str(in_size) + " |--> R_" + str(out_size))
    except Exception as e:
        interface_print("MODEL IMPORT FAILED", red=True)
        interface_print("underlining exception was: ")
        interface_print(str(e), red=True)


def export_model(file_path):
    """
    :param file_path: directory to which to save the newly learned model
    :return: no return value given
    """
    global running_model
    if file_path[-3:] != ".h5":
        single_line_print("ERROR: file of '.h5' format was expected, MODEL NOT SAVED", red=True)
        return
    try:
        interface_print('exporting model... ')
        nw.export_network(file_path, running_model)
        interface_print('done')
    except Exception as e:
        interface_print("MODEL EXPORT FAILED", red=True)
        interface_print("underlining exception was: ")
        interface_print(str(e), red=True)


def get_model_info():
    global running_model
    running_model.summary()


def create_new_model():

# %% define model architecture (play with this)

model = Sequential()
model.add(Dense(32, activation='relu', input_dim=X_train.shape[1]))
model.add(Dropout(0.1))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.05))
model.add(Dense(2048, activation='linear'))

model.summary()




# %%

# play with these parameters and see what works
batch_size = 126
epochs = 100
learning_rate = 5e-2

# reduce learning rate when no improvement are made
optim = optimizers.Adam(lr=learning_rate, beta_1=0.90, beta_2=0.999, epsilon=None, decay=1e-6, amsgrad=False)
model.compile(loss='mse', optimizer=optim, metrics=['mse'])

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=10, min_lr=0.001)

callbacks = [EarlyStopping(monitor='val_loss', patience=20),
             ModelCheckpoint('test_model', monitor='val_loss', verbose=1, save_best_only=True), reduce_lr]

history = model.fit(X_train, y_train[:, 1:].astype(float),  # skip first column since it contains the image_id
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(X_val, y_val[:, 1:].astype(float)),
                    shuffle=True,
                    callbacks=callbacks)
model.save(data_read_home + 'simple_model.h5')
# Score trained model (note that validation loss is actually the same as the mse
scores = model.evaluate(X_val, y_val[:, 1:].astype(float), verbose=1)
print('Validation loss:', scores[0])
print('Validation mse:', scores[1])

# %% score trained model and visualize

train_scores = model.evaluate(X_train, y_train[:, 1:].astype(float), verbose=1)
val_scores = model.evaluate(X_val, y_val[:, 1:].astype(float), verbose=1)

print('Training loss:', train_scores[0], ', training mse: ', train_scores[1])
print('Validation loss:', val_scores[0], ', validation mse: ', val_scores[1])

real_epochs = len(history.history['mse'])

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(np.arange(1, real_epochs + 1, 1), history.history['mse'], 'g-', label='training')
plt.plot(np.arange(1, real_epochs + 1, 1), history.history['val_mse'], 'r-', label='validation')
plt.xlabel("Epochs")
plt.ylabel("Mse")
plt.ylim([0, 0.4])
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(np.arange(1, real_epochs + 1, 1), history.history['loss'], 'g-', label='training')
plt.plot(np.arange(1, real_epochs + 1, 1), history.history['val_loss'], 'r-', label='validation')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.ylim([0, 0.4])
plt.legend()

plt.show()

# %% make predictions
predictions_val = model.predict(X_val)


# %% TODO: 1) Make functins that checks for each predictions which images are closest (rank them)


def rank_images(true_label, predictions, scoring_function='mse'):
    """

    :param true_label:
    :param predictions:
    :param scoring_function:
    :return:
    """


"""
    ranking = {}
    for i in range(len(true_label)):
        scores_ = {}
        for j in range(len(true_label)):
            if scoring_function == 'cosine':
                score = spatial.distance.cosine(predictions[i, :], true_label[j, 1:].astype(float))
            elif scoring_function == 'mse':
                # element wise mse
                score = ((predictions[i, :] - true_label[j, 1:].astype(float)) ** 2).mean(axis=None)
            else:
                print("metric not available, available metrics include mse and cosine")
            scores_[true_label[j, 0]] = score
        ranking[true_label[i, 0]] = OrderedDict(sorted(scores_.items(), key=lambda x: x[1]))
    return ranking


# %%
out = rank_images(true_label=y_val, predictions=predictions_val, scoring_function='cosine')
"""
