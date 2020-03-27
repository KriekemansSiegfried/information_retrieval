import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


from include.networks import network
from sklearn.feature_extraction.text import CountVectorizer

sns.set()
# keras
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from include.main import ranking

# own functions
from include.experimental.clean_data import clean_data
from include.experimental.split_data import split_data
from include.experimental.compute_performance import compute_performance

# %% load data
# data read home directory, this differs on some of our computers
# data_read_home = "include/data/"  # |--> for Pieter-Jan and Siegfried
data_read_home = "../data/"  # |--> for Giel

# read in train train/validation/test indices
# split according to https://github.com/BryanPlummer/flickr30k_entities
train_idx = pd.read_csv(data_read_home + 'train_idx.txt', dtype=object, header=None).values.flatten()
val_idx = pd.read_csv(data_read_home + 'val_idx.txt', dtype=object, header=None).values.flatten()
test_idx = pd.read_csv(data_read_home + 'test_idx.txt', dtype=object, header=None).values.flatten()

# split image data in train/validation/test set
# load first the whole image dataset in
df_image = pd.read_csv(data_read_home + "image_features.csv", sep=" ", header=None)
# remove .jpg from first row
df_image.iloc[:, 0] = df_image.iloc[:, 0].str.split('.').str[0]
# subset image data based idx
df_image_train = df_image[df_image.iloc[:, 0].isin(train_idx.tolist())]
df_image_val = df_image[df_image.iloc[:, 0].isin(val_idx.tolist())]
df_image_test = df_image[df_image.iloc[:, 0].isin(test_idx.tolist())]

print(f"shape training data image: {df_image_train.shape}")
print(f"shape validation data image: {df_image_val.shape}")
print(f"shape test data image: {df_image_test.shape}")

# %%
# read in caption data
doc = clean_data.load_doc(data_read_home + '/results_20130124.token', encoding="utf8")
# parse descriptions (using all descriptions)
descriptions = clean_data.load_descriptions(doc, first_description_only=False)
print('Loaded: %d ' % len(descriptions))

# clean descriptions
clean_data.clean_descriptions(descriptions, min_word_length=3)
# summarize vocabulary (still additional cleaning needed)
all_tokens = ' '.join(descriptions.values()).split()
vocabulary = set(all_tokens)
print('Vocabulary Size: %d' % len(vocabulary))
# save cleaned descriptions (
# save_doc(descriptions, data_read_home+'/descriptions.txt')
# descriptions = load_clean_descriptions('include/data/descriptions.txt')
# print('Loaded %d' % (len(descriptions)))

# %% split cleaned description in training/validation/test set
train_dic, val_dic, test_dic = split_data.train_val_test_set_desc(descriptions,
                                                                  train_idx, val_idx, test_idx)

#%%
# !! Read the API of scikit learn !!
# https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html

vectorizer = CountVectorizer(stop_words='english', min_df=1, max_df=100)
# fit on training data (descriptions)
vectorizer.fit(train_dic.values())
print("saving vectorizer")
filename = '../data/vectorizer_model.sav'
joblib.dump(vectorizer, filename)

print(len(vectorizer.vocabulary_))
# transform descriptions (based on the fit from the training data)
train_captions = vectorizer.transform(train_dic.values())
val_captions = vectorizer.transform(val_dic.values())
test_captions = vectorizer.transform(test_dic.values())

print(f"Dimension training caption: {train_captions.shape}")
print(f"Dimension validation caption: {val_captions.shape}")
print(f"Dimension test caption: {test_captions.shape}")

# %% subset captions and image to start with few examples set everything to maximum

# training
num_samples_train = 2000  # max 29783
X_train = train_captions[0:num_samples_train, :].todense()
y_train = df_image_train.iloc[0:num_samples_train, :].values
# validation
num_samples_val = 200  # max 1000
X_val = val_captions[0:num_samples_val, :].todense()
y_val = df_image_val.iloc[0:num_samples_val, :].values

# test
num_samples_test = 200  # max 1000
X_test = test_captions[0:num_samples_test, :].todense()
y_test = df_image_test.iloc[0:num_samples_test, :].values

print(f'Size train X: {X_train.shape}, train y labels {y_train.shape}')
print(f'Size validation X: {X_val.shape}, validation y labels {y_val.shape}')
print(f'Size test X: {X_test.shape}, validation y labels {y_test.shape}')


# ------ creating a new model -------
# define the number of intermediate layers and their size in a list
# further, define input dimension, and define custom optimizer
# finally, create model
layers = [3096]
input_dim = X_train.shape[1]
custom_optimizer = optimizers.Adam(lr=5e-3, beta_1=0.90, beta_2=0.999, epsilon=None, decay=1e-6, amsgrad=False)
model = network.get_network(32, layers, 2048, input_dim=input_dim, output_activation='linear',
                            loss='mse', optimizer=custom_optimizer, metrics=['mse'])


reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=10, min_lr=0.001)

filepath = "simple_model.hdf5"
callbacks = [EarlyStopping(monitor='val_loss', patience=20),
             ModelCheckpoint(filepath=filepath, monitor='val_loss',
                             verbose=1, save_best_only=True, mode='max'), reduce_lr]

batch_size = 256
epochs = 100
history = model.fit(X_train, y_train[:, 1:].astype(float),  # skip first column since it contains the image_id
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(X_val, y_val[:, 1:].astype(float)),
                    shuffle=True,
                    callbacks=callbacks)

# Score trained model (note that validation loss is actually the same as the mse
scores = model.evaluate(X_val, y_val[:, 1:].astype(float), verbose=1)
print('Validation loss:', scores[0])
print('Validation mse:', scores[1])

# %% score trained model and visualize

train_scores = model.evaluate(X_train, y_train[:, 1:].astype(float), verbose=1)
val_scores = model.evaluate(X_val, y_val[:, 1:].astype(float), verbose=1)

print("saving model")
network.export_network('../data/simple_model.h5', model)
print("model saved")

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

# compute_performance
# out = compute_performance.rank_images(true_label=y_train, predictions=predictions, scoring_function='mse', k=10,
#                                       verbose=True)
out = ranking.rank_images(y_train, predictions, id_included=True)
average_precision = compute_performance.comput_average_precision(out)
average_precision.mean()
