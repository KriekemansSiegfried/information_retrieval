print('performing imports... ', end='', flush=True)
from collections import OrderedDict
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
# data_read_home = "include/data/"          # |--> for Pieter-Jan and Siegfried
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
print("done", flush=True)

# %% subset captions and image to start with few examples
num_samples = 5000
X_captions_subset = df_captions[0:num_samples, :][::5].todense().astype(float)
y_image_subset = df_image.iloc[0:int(num_samples / 5), :].values

# make train and validation set (test set is for later once we have found good parameters)
val_size = round(num_samples / 5 * 0.25)
X_train, X_val, y_train, y_val = train_test_split(X_captions_subset,
                                                  y_image_subset, test_size=val_size)

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
batch_size = 512
epochs = 100
learning_rate = 5e-3

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
        # save id and scores in ascending (score) order
        ranking[true_label[i, 0]] = OrderedDict(sorted(scores_.items(), key=lambda x: x[1]))
    return ranking


# %%
out = rank_images(true_label=y_val, predictions=predictions_val, scoring_function='cosine')
