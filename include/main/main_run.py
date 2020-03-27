import joblib
import matplotlib.pyplot as plt
import matplotlib.image as mp_img
import numpy as np
import pandas as pd
import seaborn as sns


from include.networks import network
from sklearn.feature_extraction.text import CountVectorizer
from include.io import import_images as io
from include.main import ranking
from include.networks.network import import_network
from include.experimental.clean_data import clean_data
from include.experimental.split_data import split_data
image_file_directory = '../data/flickr30k-images/'
image_csv_directory = '../data/image_features.csv'
path_to_vectorizer = '../data/vectorizer_model.sav'
# path_to_simple_model = '../experimental/simple_model.hdf5'
path_to_simple_model = '../data/simple_model.h5'

model = import_network(path_to_simple_model)
image_array = None
vectorizer = None
data_read_home = '../data/'
#caption_string = 'Two young guys with shaggy hair look at their hands while hanging out in the yard'
caption_string = 'young guys with shaggy hair look their hands while hanging yard white males outside near many ' \
                 'bushes green shirts standing blue shirt garden friends enjoy time spent together'

"""
def build_vectorizer():
    global vectorizer
    train_idx = pd.read_csv(data_read_home + 'train_idx.txt', dtype=object, header=None).values.flatten()
    val_idx = pd.read_csv(data_read_home + 'val_idx.txt', dtype=object, header=None).values.flatten()
    test_idx = pd.read_csv(data_read_home + 'test_idx.txt', dtype=object, header=None).values.flatten()

    df_image = pd.read_csv(data_read_home + "image_features.csv", sep=" ", header=None)
    df_image.iloc[:, 0] = df_image.iloc[:, 0].str.split('.').str[0]
    df_image_train = df_image[df_image.iloc[:, 0].isin(train_idx.tolist())]
    doc = clean_data.load_doc(data_read_home + '/results_20130124.token', encoding="utf8")
    descriptions = clean_data.load_descriptions(doc, first_description_only=False)
    clean_data.clean_descriptions(descriptions, min_word_length=3)
    # summarize vocabulary (still additional cleaning needed)
    all_tokens = ' '.join(descriptions.values()).split()
    vocabulary = set(all_tokens)
    train_dic, val_dic, test_dic = split_data.train_val_test_set_desc(descriptions,
                                                                      train_idx, val_idx, test_idx)
    vectorizer = CountVectorizer(stop_words='english', min_df=1, max_df=100)
    # fit on training data (descriptions)
    vectorizer.fit(train_dic.values())
"""

def set_image_file_directory(file_path):
    global image_file_directory
    image_file_directory = file_path


def enter_caption(new_caption_string):
    global caption_string
    caption_string = new_caption_string


def caption_to_caption_feature(caption=caption_string):
    global vectorizer, caption_string
    if vectorizer is None:
        load_caption_feature_translator(path_to_vectorizer)
        print('No vectorizer was set. Default vectorizer loaded from ' + path_to_vectorizer)
    descriptions = clean_data.load_descriptions(caption, first_description_only=False)
    caption_feature = vectorizer.transform(descriptions)
    return caption_feature


def load_caption_feature_translator(vectorizer_file_path=path_to_vectorizer):
    global vectorizer
    vectorizer = joblib.load(vectorizer_file_path)


def load_network(file_path=path_to_simple_model):
    global model
    model = import_network(file_path)


def caption_to_image_feature(caption=caption_string):
    global caption_string, model
    if model is None:
        load_network()
        print('No model was loaded yet, default model has been loaded')
    caption_feature = caption_to_caption_feature(caption).todense()
    print(caption_feature)
    image_feature = model.predict([caption_feature])
    return image_feature


def import_images(file_path=image_csv_directory):
    global image_array
    image_array = io.import_images_as_ndarray(file_path)


def get_most_relevant_image(image_feature, number_or_images=10):
    if image_array is None:
        import_images()
        print('No image set was imported, the default will be imported now')
    # images_list = ranking.rank_efficient(image_feature, image_array, id_included=True, k=number_or_images)

    # print('type of images_list:', end='', flush=True)
    # print(type(image_array), flush=True)
    # print('shape of images_list:', end='', flush=True)
    # print(image_array.shape, flush=True)
    # print('type of predictions:', end='', flush=True)
    # print(type(image_feature), flush=True)
    # print('shape of predictions:', end='', flush=True)
    # print(image_feature.shape, flush=True)
    #
    # print('type of np.array(predictions):', end='', flush=True)
    # print(type(np.array(image_feature)), flush=True)
    # print('shape of np.array(predictions):', end='', flush=True)
    # print(np.array(image_feature).shape, flush=True)

    images_list = ranking.rank_images(image_array, image_feature, k=number_or_images)
    return images_list


def run_search_engine(input_caption=caption_string):
    img_caption_feature = np.array(caption_to_image_feature(input_caption))
    print("ranking results")
    search_results = get_most_relevant_image(img_caption_feature)
    return search_results


def open_image(file_path, is_name_only=True):
    if not is_name_only:
        img = mp_img.imread(file_path)
    else:
        img = mp_img.imread(image_file_directory + file_path)
    plt.imshow(img)
    plt.show()


# general script
print("importing images")
import_images()
print("loading vectorizer")
load_caption_feature_translator()
# build_vectorizer()
print("running search engine")
search_result = run_search_engine()
print(search_result)
#open_image(search_result[0, 0])


