import joblib
import matplotlib.pyplot as plt
import matplotlib.image as mp_img
from include.io import import_images as io
from include.main import ranking
from include.networks.network import import_network
image_file_directory = '../data/flickr30k-images/'
image_csv_directory = '../data/image_features.csv'
path_to_vectorizer = '../data/vectorizer_model.sav'
path_to_simple_model = '../data/simple_model.h5'
network = None
image_array = None
vectorizer = None

caption_string = 'Two young guys with shaggy hair look at their hands while hanging out in the yard'


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
    caption_feature = vectorizer.transform([caption])
    return caption_feature


def load_caption_feature_translator(vectorizer_file_path=path_to_vectorizer):
    global vectorizer
    vectorizer = joblib.load(vectorizer_file_path)


def load_network(file_path=path_to_simple_model):
    global network
    network = import_network(file_path)


def caption_to_image_feature(caption=caption_string):
    global caption_string, model
    if model is None:
        load_network()
        print('No model was loaded yet, default model has been loaded')
    caption_feature = caption_to_caption_feature(caption)
    image_feature = model.predict(caption_feature.todense())
    return image_feature[0]


def import_images(file_path=image_csv_directory):
    global image_array
    image_array = io.import_images_as_ndarray(file_path)


def get_most_relevant_image(image_feature, number_or_images=10):
    if image_array is None:
        import_images()
        print('No image set was imported, the default will be imported now')
    images_list = ranking.rank_images(image_feature, image_array, id_included=True, k=number_or_images)
    return images_list


def run_search_engine(input_caption=caption_string):
    caption_feature = caption_to_caption_feature(input_caption)
    imp_caption_feature = caption_to_image_feature(caption_feature)
    search_results = get_most_relevant_image(imp_caption_feature)
    return search_results


def open_image(file_path, is_name_only=True):
    if not is_name_only:
        img = mp_img.imread(file_path)
    else:
        img = mp_img.imread(image_file_directory + file_path)
    plt.imshow(img)
    plt.show()


# general script
import_images()
load_network()
load_caption_feature_translator()
search_result = run_search_engine()
open_image(search_result[0, 0])
print(search_result)


