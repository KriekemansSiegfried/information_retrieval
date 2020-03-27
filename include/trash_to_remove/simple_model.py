print('performing imports... ', end='', flush=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sparse
import seaborn as sns
from scipy import spatial

sns.set()
# keras
from sklearn.model_selection import train_test_split
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

print('done', flush=True)

# data read home directory, this differs on some of our computers
# data_read_home = "include/data/"  # |--> for Pieter-Jan and Siegfried
data_read_home = "../data/"  # |--> for Giel

# %% load data
print("loading 'caption_features.npz'... ", end='', flush=True)
df_captions = sparse.load_npz(data_read_home + "caption_features.npz")
print("done", flush=True)
# if you want to go to the uncompressed format
# df_captions_uncomp = df_captions.todense()

# images (normal format) (this is in pandas dataframe format) (31782, 2049)
print("loading 'image_features.csv'... ", end='', flush=True)
df_image = pd.read_csv(data_read_home + "image_features.csv",
                       sep=" ", header=None)
# remove .jpg from first row
df_image.iloc[:, 0] = df_image.iloc[:, 0].str.split('.').str[0]
print("done", flush=True)

# %% read in train train/validation/test indices
# split according to https://github.com/BryanPlummer/flickr30k_entities
train_idx = pd.read_csv(data_read_home + 'train_idx.txt', dtype=object, header=None).values.flatten()
val_idx = pd.read_csv(data_read_home + 'val_idx.txt', dtype=object, header=None).values.flatten()
test_idx = pd.read_csv(data_read_home + 'test_idx.txt', dtype=object, header=None).values.flatten()

# subset image data based idx
df_image_train = df_image[df_image.iloc[:, 0].isin(train_idx.tolist())]
df_image_val = df_image[df_image.iloc[:, 0].isin(val_idx.tolist())]
df_image_test = df_image[df_image.iloc[:, 0].isin(test_idx.tolist())]

print(f"shape training data image: {df_image_train.shape}")
print(f"shape validation data image: {df_image_val.shape}")
print(f"shape test data image: {df_image_test.shape}")

# %%


# %% subset captions and image to start with few examples
num_samples = 500
X_captions_subset = df_captions[0:num_samples, :].todense().astype(float)
y_image_subset = df_image_train.iloc[0:int(num_samples), :].values

# make train and validation set (test set is for later once we have found good parameters)
val_size = round(num_samples * 0.25)
X_train, X_val, y_train, y_val = train_test_split(X_captions_subset,
                                                  y_image_subset, test_size=val_size)
# print("input type = " + str(type(X_train)))
# print("input type = " + str(type(X_train[0])))
# print(X_train[0])
print(f'Size train X: {X_train.shape}, train y labels {y_train.shape}')
print(f'Size validation X: {X_val.shape}, validation y labels {y_val.shape}')

# %% define model architecture (play with this)

model = Sequential()
model.add(Dense(32, activation='relu', input_dim=X_train.shape[1]))
# model.add(Dropout(0.1))
model.add(Dense(4096, activation='relu'))
# model.add(Dropout(0.05))
model.add(Dense(2048, activation='linear'))

model.summary()

# %%

# play with these parameters and see what works
batch_size = 128
epochs = 100
learning_rate = 1e-3

# reduce learning rate when no improvement are made
optim = optimizers.Adam(lr=learning_rate, beta_1=0.90, beta_2=0.999, epsilon=None, decay=1e-6, amsgrad=False)
model.compile(loss='mse', optimizer=optim, metrics=['mse'])

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=10, min_lr=0.001)

filepath = "simple_model.hdf5"
callbacks = [EarlyStopping(monitor='val_loss', patience=20),
             ModelCheckpoint(filepath=filepath, monitor='val_loss',
                             verbose=1, save_best_only=True, mode='max'), reduce_lr]

history = model.fit(X_train, y_train[:, 1:].astype(float),  # skip first column since it contains the image_id
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(X_val, y_val[:, 1:].astype(float)),
                    shuffle=True,
                    callbacks=callbacks)

#model.save(data_read_home + 'simple_model.h5')
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
predictions = model.predict(X_train)


# %% TODO: 1) Make functins that checks for each predictions which images are closest (rank them)
import sys


def printProgressBar(i, max, postText):
    n_bar = 10  # size of progress bar
    j = i / max
    sys.stdout.write('\r')
    sys.stdout.write(f"[{'=' * int(n_bar * j):{n_bar}s}] {int(100 * j)}%  {postText}")
    sys.stdout.flush()


def rank_images(true_label, predictions, scoring_function='mse', k=10, verbose=True):


    """

    :param true_label:
    :param predictions:
    :param scoring_function:
    :return:
    """

    ranking = {}
    n = len(true_label)
    for i in range(n):
        printProgressBar(i, n, "Finish")
        scores_ = []
        for j in range(len(true_label)):
            if scoring_function == 'cosine':
                score = spatial.distance.cosine(predictions[i, :], true_label[j, 1:].astype(float))
            elif scoring_function == 'mse':
                # element wise mse
                score = ((predictions[i, :] - true_label[j, 1:].astype(float)) ** 2).mean(axis=None)
            else:
                print("metric not available, available metrics include mse and cosine")
            scores_.append((true_label[j, 0], score))
        # save lowest k id's and scores in ascending (score) order
        ranking[true_label[i, 0]] = (sorted(scores_, key=lambda x: x[1]))[0:k]
    return ranking


# %% rank images
out = rank_images(true_label=y_train, predictions=predictions, scoring_function='mse', k=10)


# %% compute MAPE 10

def comput_average_precision(dic):
    """
    :param dic:
    :return:
    """

    store_idx = {}
    counter = 0
    n = len(dic.items())
    for key, value in dic.items():
        counter += 1
        printProgressBar(counter, n, "Finish")
        list_keys = [item[0] for item in value]
        # check if ground true label (image_id) is in in the first k (=10) predicted labels (image_id)
        if key in list_keys:
            np_array = np.array(list_keys)
            # get indice where ground true label (image_id) == predicted label (image_id)
            item_index = np.where(np_array == key)[0][0]
            # compute the average precision
            store_idx[key] = 1 / (item_index + 1)
        else:
            # ground true label (image_id) is not in in the first k (=10) predicted labels (image_id)
            # ==> average precision = 0
            store_idx[key] = 0
    return pd.DataFrame.from_dict(store_idx, orient='index', columns=['score'])
