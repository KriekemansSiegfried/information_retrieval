import os

"""
     ___       ____      _____                ____    _
    |   |     |    |    |     |    |   _|    |       | |_      |
    |___|     |____|    |     |    |__|      |____   |   |_    |
    |    |    | |_      |     |    |   |_    |       |     |_  |
    |____|    |   |     |_____|    |     |   |____   |       |_|
   
   
   THIS FILE NO LONGER WORKS 
"""



import joblib
import matplotlib.image as mp_img
import matplotlib.pyplot as plt
import numpy as np
from include.part1.simple_model.preprocess_data import preprocessing as io, preprocessing, ranking

# from include.search_engine import ranking
from include.networks.network import import_network
#from include.part1 import data_analysis as ranking

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

image_file_directory = '../model/flickr30k-images/'
image_csv_directory = '../model/image_features.csv'
path_to_vectorizer = '../model/vectorizer_model.sav'
# path_to_simple_model = '../simple_model/simple_model.hdf5'
path_to_simple_model = '../model/simple_model.h5'

model = import_network(path_to_simple_model)
image_array = None
vectorizer = None
data_read_home = '../model/'
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
    doc = preprocess_data.load_doc(data_read_home + '/results_20130124.token', encoding="utf8")
    descriptions = preprocess_data.load_descriptions(doc, first_description_only=False)
    preprocess_data.clean_descriptions(descriptions, min_word_length=3)
    # summarize vocabulary (still additional cleaning needed)
    all_tokens = ' '.join(descriptions.values()).split()
    vocabulary = set(all_tokens)
    train_dic, val_dic, test_dic = split_data.train_val_test_set_desc(descriptions,
                                                                      train_idx, val_idx, test_idx)
    vectorizer = CountVectorizer(stop_words='english', min_df=1, max_df=100)
    # fit on training model (descriptions)
    vectorizer.fit(train_dic.values())
"""

"""---------------------------------------------------- GLOBAL FIELD SETTERS ----------------------------------------"""


def set_image_file_directory(file_path):
    """
    set the global image file directory path
    :param file_path: the newly proposed file directory path
    :return:
    """
    global image_file_directory
    image_file_directory = file_path


def set_caption(new_caption_string):
    """
    manually set the search caption to a new value
    :param new_caption_string: the new search caption
    :return: /
    """
    global caption_string
    caption_string = new_caption_string


"""---------------------------------------------------- DATA AND STRUCTURE IMPORTS ----------------------------------"""


def import_images(file_path=image_csv_directory):
    """
    imports the images
    image_array will become an n*m array, where n equals the number of images,
    and m equals ghe number of features for each image + 1 (since the first column
    will contain the image name)
    :param file_path: the file path from which to load the image features, if not set, the
                       default (global) 'image_csv_directory' -path wil be used
    :return: /
    """
    global image_array, image_csv_directory
    if file_path != image_csv_directory:
        set_image_file_directory(file_path)
    image_array = io.import_images_as_ndarray(file_path)


def load_caption_feature_translator(vectorizer_file_path=path_to_vectorizer):
    """
    loads a vectorizer, which is a translator module of type
        <class 'sklearn.feature_extraction.text.CountVectorizer'>
    used to translate string format captions into <numpy.matrix> format vectors, which
    is the used input format for the simple model
    :param vectorizer_file_path: path to the vectorizer which is to be loaded. If not given,
                                 the global field value 'path_to_vectorizer' is used
    :return: / --> global field value 'vectorizer' is set to the loaded vectorizer
    """
    global vectorizer
    vectorizer = joblib.load(vectorizer_file_path)


def load_network(file_path=path_to_simple_model):
    """
    loads the network to be used tor translate a caption_feature vector into
    an image_feature vector. A simple model format is expected
    :param file_path: the directory to the network. If not given, the global
                      field value 'path_to_simple_model' will be used
    :return: / --> global field value 'model' is set to the loaded model
    """
    global model
    model = import_network(file_path)


"""---------------------------------------------------- DATA TRANSLATION AND ANALYSIS -------------------------------"""


def caption_to_caption_feature(caption=caption_string):
    """
    translates a caption (string) into a caption-feature
    :param caption: the input caption, if not given, the global field
                    value 'caption_string' will be used
    :return: a caption feature vector in <numpy.ndarray> format
    """
    global vectorizer, caption_string
    if vectorizer is None:
        load_caption_feature_translator(path_to_vectorizer)
        print('No vectorizer was set. Default vectorizer loaded from ' + path_to_vectorizer)
    descriptions = preprocessing.load_descriptions(caption, first_description_only=False)
    caption_feature = vectorizer.transform(descriptions)
    return caption_feature


def caption_to_image_feature(caption=caption_string):
    """
    translates a caption (string) to an image_feature vector
    :param caption: the input caption. If not given, the global field value
                    'caption_string' will be used
    :return: image feature of type <numpy.ndarray>
    """
    global caption_string, model
    if model is None:
        load_network()
        print('No model was loaded yet, default model has been loaded')
    caption_feature = caption_to_caption_feature(caption).todense()
    #print(caption_feature)
    image_feature = model.predict([caption_feature])
    #print('image_feature_type: ' + str(type(image_feature)))
    return image_feature


def get_most_relevant_image(image_feature, number_or_images=10):
    """
    searches the |number_or_images| most relevant images to the
    given image_feature, present in the loaded image_array
    :param image_feature: the image feature of which most relevant is requested
    :param number_or_images: the number of most relevant images that is
                             to be returned (10 if not given)
    :return: a list of images that is most relevant to the given input feature
    """
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
    images_list = ranking.rank_images(image_array, image_feature, k=number_or_images, verbose=False, batch_sizes_equal=False)
    return images_list


def run_search_engine(input_caption=caption_string):
    """
    runs the search engine on the given input string
    :param input_caption: the input string on which to run the search engine. If not given,
                          the global field value 'caption_string' is used
    :return: a list of the 10 most relevant images to the given input caption
    """
    img_caption_feature = np.array(caption_to_image_feature(input_caption))
    #print("ranking results")
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
print('importing images ... ', end='', flush=True)
import_images()
print('done \nloading vectorizer ... ', end='', flush=True)
load_caption_feature_translator()
print('done \nloading network ... ', end='', flush=True)
load_network()
print('done \nrunning search engine ... ', end='', flush=True)
search_result = run_search_engine()
print('done \nresults: ', flush=True)
print(search_result)
#open_image(search_result[0, 0])


