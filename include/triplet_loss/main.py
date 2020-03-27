from nltk.corpus import stopwords
from numpy.linalg import norm

from include.bow import dictionary, one_hot
from include.io import import_captions, import_images
from include.triplet_loss.features import get_caption_embedding, get_image_embedding
from include.triplet_loss.load_model import load_submodels

caption_filename = 'include/data/results_20130124.token'
image_filename = 'include/data/image_features.csv'
model_weights_path = 'include/data/best_model_triplet.h5'
model_json_path = 'include/data/best_model_triplet.json'

# read in data
print('loading data')
captions = import_captions.import_captions(caption_filename)
images = import_images.import_images(image_filename)
print('loaded {} captions'.format(len(captions)))
print('loaded {} images'.format(len(images)))
bow_dict = dictionary.create_dict(captions)

# %%
# get stop words
stop_words = set(stopwords.words('english'))

# prune dictionary
bow_dict_pruned, removed_words = dictionary.prune_dict(word_dict=bow_dict,
                                                       stopwords=stop_words,
                                                       min_word_len=3,
                                                       min_freq=1,
                                                       max_freq=1000)
tokens = list(bow_dict_pruned.keys())
progress = 0
pruned_captions = []
for caption in captions:
    progress += 1
    if progress % 2500 == 0:
        print(progress)
    caption.features = one_hot.convert_to_bow(caption, tokens)
    pruned_captions.append(caption)
    if (progress == 5000):
        break
# %%

caption_model, image_model = load_submodels(model_path=model_json_path, weights_path=model_weights_path)

# test caption ranking
caption = captions[0]
print('caption -> {}'.format(caption.tokens))

caption_embedding = get_caption_embedding(caption, caption_model)


def distance(feature_1, feature_2):
    return norm(feature_1 - feature_2, ord=2)


# %%

images = images[:2000]
images = list(
    map(lambda image: (image.image_name, distance(caption_embedding, get_image_embedding(image, image_model))), images))

print('images -> {}'.format(images))

images.sort(key=lambda image: image[1])

print('sorted -> {}'.format(images))
