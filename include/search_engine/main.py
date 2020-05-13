# %% import libraries

import numpy as np
import seaborn as sns
from include.bow import dictionary, one_hot
from include.io import import_captions, import_images
# to quickly reload functions
from include.part1.data_analysis import convert_to_triplet_dataset
from nltk.corpus import stopwords
# custom defined functions
from tensorflow_core.python.keras.utils.vis_utils import plot_model

# style seaborn for plotting
# %matplotlib qt5 (for interactive plotting: run in the python console)
from include.networks.network import get_network_triplet_loss
# from include.training.dataset import convert_to_triplet_dataset
from include.util.util import get_pairs_images, make_dict

# interactive plotting
# %matplotlib qt5

sns.set()
# print numpy arrays in full
# np.set_printoptions(threshold=sys.maxsize)


# %%  import model

# caption_filename = '/home/kriekemans/KUL/information_retrieval/dataset/results_20130124.token'
# image_filename = '/home/kriekemans/KUL/information_retrieval/dataset/image_features.csv'

caption_filename = 'include/model/results_20130124.token'
image_filename = 'include/model/image_features.csv'
weights_path = 'include/model/best_model_triplet.h5'
model_json_path = 'include/model/best_model_triplet.json'

# # read in model
captions = import_captions.import_captions(caption_filename)
images = import_images.import_images(image_filename)

print('loaded {} captions'.format(len(captions)))
print('loaded {} images'.format(len(images)))

# %% create captions to bow dictionary
bow_dict = dictionary.create_dict(captions)


# %%
# get stop words
stop_words = set(stopwords.words('english'))

# prune dictionary
bow_dict_pruned, removed_words = dictionary.prune_dict(word_dict=bow_dict,
                                                       stopwords=stop_words,
                                                       min_word_len=3,
                                                       min_freq=0,
                                                       max_freq=1000)

# have a look again at the most frequent words from the updated dictionary
# _ = fw.rank_word_freq(dic=bow_dict_pruned, n=20, ascending=False, visualize=True)

# have a look at the removed words
# _ = fw.rank_word_freq(dic=removed_words, n=20, ascending=False, visualize=True)

# %% # one hot encode
tokens = list(bow_dict_pruned.keys())

print('converting caption features')

caption_feature_size = len(tokens)

progress = 0
pruned_captions = []
for caption in captions:
    progress += 1
    if progress % 2500 == 0:
        print(progress)
    # efficiently store sparse matrix
    # see https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html
    caption.features = one_hot.convert_to_bow(caption, tokens)
    pruned_captions.append(caption)
    if progress == 2500:
        break
# %%

captions = pruned_captions

print('features converted')
#
print('creating triplet dict')
pair_dict = make_dict(images, captions)
print('creating triplets')
pairs = get_pairs_images(pair_dict)

# %%
print('pairs created')
print('creating dataset with labels')
dataset, labels = convert_to_triplet_dataset(pairs)
print('dataset created')

# %%
print('network loading')
from tensorflow.keras import optimizers

custom_optimizer = optimizers.Adam(lr=1e-4, beta_1=0.90, beta_2=0.999, epsilon=None, decay=1e-6, amsgrad=False)
network = get_network_triplet_loss(caption_feature_size, len(images[0].features), 512)
print('network loaded')
model_json = network.to_json()
with open(model_json_path, 'w') as json_file:
    json_file.write(model_json)

plot_model(network, to_file='model.png', show_shapes=True, show_layer_names=True)

# %%
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau

reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2,
                              patience=5, min_lr=0.00001)

callbacks = [EarlyStopping(monitor='loss', patience=10),
             ModelCheckpoint(filepath=weights_path, monitor='loss',
                             verbose=1, save_best_only=True, mode='min'), reduce_lr]

real_epochs = 15
batch_size = 128
network.fit(dataset, labels, epochs=real_epochs, batch_size=batch_size, callbacks=callbacks)

print('network fit done!')

# %% plot training and validation loss
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
plt.plot(np.arange(1, 30 + 1, 1), network.history.history['loss'], 'g-', label='training')
# plt.plot(np.arange(1, real_epochs + 1, 1), history.history['val_mse'], 'r-', label='validation')
plt.xlabel("Epochs")
plt.ylabel("Triplet loss")
plt.ylim([0, 10])
plt.legend()
plt.show()
