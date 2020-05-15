# %%
# data preprocessing
import json
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import joblib

# visualize
import matplotlib.pyplot as plt
import seaborn as sns

# tensorflow
from include.networks import network
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.python.keras.utils.vis_utils import plot_model

# own modules
from include.part1.triplet_loss.load_model import get_embedding
from include.part1.triplet_loss.load_model import load_model
from include.preprocess_data import preprocessing
from include.ranking import ranking as ranking

# TODO:
#  - make generator for sparse matrices: https://stackoverflow.com/questions/37609892/keras-sparse-matrix-issue
#  - compute MAP@10 on train and test model: DONE https://towardsdatascience.com/breaking-down-mean-average-precision-map-ae462f623a52
#  - add documentation: DONE
#  - try to regularize:
#  - play wit hyperparameters:
#  - clean code and refactor: DONE
#  - structureize project: DONE

# %% GLOBAL VARIABLES (indicated in CAPITAL letters)
PATH = "include/input/"
SAVE_FIGURES = 'include/output/figures/triplet_loss/'
SAVE_BOW_MODEL = 'include/output/model/triplet_loss/caption_bow_model.pkl'
MODEL_JSON_PATH = 'include/output/model/triplet_loss/best_model.json'
MODEL_WEIGHTS_PATH = 'include/output/model/triplet_loss/best_model.h5'
sns.set()
# %% read in image model
image_train, image_val, image_test = preprocessing.read_split_images(path=PATH)
#
# %% read in caption model and split in train, validation and test set and save it
caption_train, caption_val, caption_test = preprocessing.read_split_captions(
    path=PATH, document='results_20130124.token', encoding="utf8", dir="include/output/data")

# %% in case you already have ran the cel above once before and don't want to run it over and over
# train
caption_train = json.load(open('include/output/data/train.json', 'r'))
# val
caption_val = json.load(open('include/output/data/val.json', 'r'))
# test
caption_test = json.load(open('include/output/data/test.json', 'r'))

# %% clean captions (don't run this more than once or
# you will prune your caption dictionary even further as it has the same variable name)

# experiment with it: my experience: seems to work better when training
stemming = True
caption_train = preprocessing.clean_descriptions(
    descriptions=caption_train, min_word_length=2, stem=stemming, unique_only=False
)
caption_val = preprocessing.clean_descriptions(
    descriptions=caption_val, min_word_length=2, stem=stemming, unique_only=False
)
caption_test = preprocessing.clean_descriptions(
    descriptions=caption_test, min_word_length=2, stem=stemming, unique_only=False
)

# %% convert to bow
c_vec = CountVectorizer(stop_words='english', min_df=1, max_df=100000)
# fit on training model (descriptions)

c_vec.fit(caption_train.values())
print(f"Size vocabulary: {len(c_vec.vocabulary_)}")
# transform on train/val/test model
caption_train_bow = [list(caption_train.keys()), c_vec.transform(caption_train.values())]
caption_val_bow = [list(caption_val.keys()), c_vec.transform(caption_val.values())]
caption_test_bow = [list(caption_test.keys()), c_vec.transform(caption_test.values())]

# save vectorizer for later (search engine)
joblib.dump(c_vec, SAVE_BOW_MODEL)
# load your model (just as a test)
# c_vec = joblib.load(SAVE_BOW_MODEL)
# %% train
n_images_train = 29783  # whole dataset:  n_images_train = 29783
caption_id_train, dataset_train, labels_train = preprocessing.convert_to_triplet_dataset(
    captions=caption_train_bow, images=image_train, captions_k=5,
    p=100, n_row=n_images_train, todense=True
)

# %% validation
n_images_val = 1000  # whole dataset
caption_id_val, dataset_val, labels_val = preprocessing.convert_to_triplet_dataset(
    captions=caption_val_bow, images=image_val, captions_k=5, p=25, n_row=n_images_val, todense=True
)

# %% test
n_images_test = 1000  # whole dataset
caption_id_test, dataset_test, labels_test = preprocessing.convert_to_triplet_dataset(
    captions=caption_test_bow, images=image_test, captions_k=5, p=25, n_row=n_images_test, todense=True
)

# %%
print('network loading')

caption_feature_size = dataset_train[0].shape[1]
image_feature_size = dataset_train[2].shape[1]
custom_optimizer = optimizers.Adam(
    lr=1e-4, beta_1=0.90, beta_2=0.999, epsilon=None, decay=1e-6, amsgrad=False
)
model = network.get_network_triplet_loss(
    caption_size=caption_feature_size, image_size=image_feature_size,
    embedding_size=512, triplet_margin=300, optimizer=custom_optimizer
)
# %%
# save network and plot
model_json = model.to_json()
with open(MODEL_JSON_PATH, 'w') as json_file:
    json_file.write(model_json)

plot_model(model, to_file=SAVE_FIGURES + 'architecture.png',
           show_shapes=True, show_layer_names=True)

# %%
reduce_lr = ReduceLROnPlateau(
    monitor='loss', factor=0.3, patience=1, min_lr=0.00001
)

callbacks = [EarlyStopping(monitor='val_loss', patience=10),
             ModelCheckpoint(filepath=MODEL_WEIGHTS_PATH, monitor='val_loss',
                             verbose=1, save_best_only=True, mode='min'), reduce_lr]

# %%
real_epochs = 25
batch_size = 512
model.fit(
    x=dataset_train,
    y=labels_train,
    epochs=real_epochs,
    batch_size=batch_size,
    validation_data=(dataset_val, labels_val),
    callbacks=callbacks,
)

print('network fit done!')

# %% visualize training
num_e_done = len(model.history.history['loss'])

plt.figure(figsize=(10, 8))
plt.plot(np.arange(1, num_e_done + 1, 1), model.history.history['loss'], 'g-', label='training')
plt.plot(np.arange(1, num_e_done + 1, 1), model.history.history['val_loss'], 'r-', label='validation')
plt.xlabel("Epochs")
plt.ylabel("Triplet loss")
plt.ylim([0, 100])
plt.legend()
plt.show()
# save figure
plt.savefig(SAVE_FIGURES + "training.png")

# %% Import best trained caption and image model
caption_model, image_model = load_model.load_submodels(
    model_path=MODEL_JSON_PATH, weights_path=MODEL_WEIGHTS_PATH
)

# %%
# -----------------------------------------
# Test caption ranking on TEST DATA
# ----------------------------------------
# index 1 are the positive captions
# reshape=True if you only have one vector

# 1) make predictions
caption = dataset_test[1]
caption_embedding = get_embedding.get_caption_embedding(caption, caption_model, reshape=False)

# index 2 are the image feature
# only make a prediction for every 5th, image vectors don't change, only captions do
image = dataset_test[2][::5]
image_embedding = get_embedding.get_image_embedding(image, image_model, reshape=False)

# %% 2 a) compute ranking images
image_id = image_test.iloc[0:n_images_test, 0].values
caption_id = np.array(caption_id_test[1][0:n_images_test * 5])

ranking_images = ranking.rank_embedding(
    caption_embed=caption_embedding,
    caption_id=caption_id,
    image_embed=image_embedding,
    image_id=image_id,
    retrieve="images",
    distance_metric="L2",
    k=10,
    add_correct_id=True
)

# %% 2 b) compute ranking captions
ranking_captions = ranking.rank_embedding(
    caption_embed=caption_embedding,
    caption_id=caption_id,
    image_embed=image_embedding,
    image_id=image_id,
    retrieve="captions",
    distance_metric="L2",
    k=10,
    add_correct_id=True
)
#
# %% 3 a) compute MAP@10 images
average_precision_images = ranking.average_precision(ranking_images, gtp=1)
print(f"{average_precision_images.head()}")
print(f"Mean average precision @10 is: {round(average_precision_images.mean()[0] * 100, 4)} %")

# %% 3 b) compute MAP@10 captions
average_precision_captions = ranking.average_precision(ranking_captions, gtp=5)
print(f"{average_precision_captions.head()}")
print(f"Mean average precision @10 is: {round(average_precision_captions.mean()[0] * 100, 4)} %")
