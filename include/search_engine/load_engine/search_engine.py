# data transformations
import numpy as np
import joblib
import pandas as pd
from numpy.linalg import norm
from scipy.sparse import isspmatrix
import copy
from math import ceil
import sys
# save
import os
import pickle

# visualization
import cv2
import matplotlib.pyplot as plt


# own functions
from include.part1.triplet_loss.load_model import load_model
from include.preprocess_data import preprocessing
from include.util.util import print_progress_bar
from include.util.util import get_batch


class SearchEngine:

    def __init__(self,
                 mode="triplet_loss",
                 clean=True,
                 min_word_length=2,
                 stem=True,
                 unique_only=False,
                 k=10,
                 path_transformer='include/output/model/triplet_loss/caption_bow_model.pkl',
                 model_path='include/output/model/triplet_loss/best_model.json',
                 weights_path='include/output/model/triplet_loss/best_model.h5',
                 database_images_path="include/output/data/triplet_loss/database_images/database_images.dat",
                 database_captions_path="include/output/data/triplet_loss/database_captions/database_captions.dat",
                 image_dir="include/input/flickr30k-images/"):

        super().__init__()
        self.mode = mode
        self.clean = clean
        self.min_word_length = min_word_length
        self.stem = stem
        self.unique_only = unique_only
        self.k = k
        self.path_transformer = path_transformer
        self.model_path = model_path
        self.weights_path = weights_path
        self.database_images_path = database_images_path
        self.database_captions_path = database_captions_path
        self.image_dir = image_dir
        self.new = None  # new caption or new image vector
        self.new_id = None
        self.image_model = None
        self.caption_model = None
        self.caption_transformer = None
        self.database = None
        self.new_embedding = None
        self.ranking = None
        self.distance_metric = None
        self.figsize = None
        self.title_fontsize = None
        self.title_y = None
        self.title_x = None
        self.nrows = None
        self.ncols = None


    def load_models_triplet_loss(self, verbose=True):
        if verbose:
            print("TRIPLET LOSS (PART 1): Loading models")

        try:
            self.caption_model, self.image_model = load_model.load_submodels(
                model_path=self.model_path, weights_path=self.weights_path
            )
        except ValueError:
            print("Model triptlet loss not found, provide a valid path")

    def load_caption_transformer_triplet_loss(self, verbose=True):

        if verbose:
            print("TRIPLET LOSS (PART 1): Loading caption transformer")
        try:
            self.caption_transformer = joblib.load(self.path_transformer)
        except ValueError:
            print("transformer model triplet loss not found, provide a valid path")

    def load_cross_modal_retrieval(self, verbose=True):
        if verbose:
            print("CROSS MODAL (PART 2): Loading models")

        try:
            pass
            # TODO: load modal part 2
            # self.caption_model, self.image_model =
        except ValueError:
            print("transformer cross_modal not found, provide a valid path")

    def load_caption_transformer_cross_modal(self, verbose=True):
        if verbose:
            print("CROSS MODAL (PART 1): Loading caption transformer")
        try:
            self.caption_transformer = joblib.load(self.path_transformer)
        except ValueError:
            print("transformer model triplet loss not found, provide a valid path")

    def load_database_images(self):
        try:
            database = joblib.load(self.database_images_path)
        except ValueError:
            print("image database could not be loaded, provide a valid path")
        return database

    def load_database_captions(self):
        try:
            database = joblib.load(self.database_captions_path)
        except ValueError:
            print("caption database could not be loaded, provide a valid path")
        return database

    def embed_new_caption(self, new_caption_id="361092202.jpg#4"):

        #  add functionality to load caption id as a new caption
        # check if there is a caption or image id provided

        # A) image_id provided and no new_caption
        if self.new is None and new_caption_id is not None:
            self.new_id = new_caption_id
            try:
                all_captions = preprocessing.load_captions()
            except ValueError:
                print("captions could not been loaded")
            # dictionary format
            new_caption = {new_caption_id: all_captions[new_caption_id]}
            self.new = copy.deepcopy(new_caption)

        # B) caption provided
        elif self.new is not None:

            # change to dictionary format
            if isinstance(self.new, str):
                new_caption = {"New Caption": self.new}
                self.new = copy.deepcopy(new_caption)
            else:
                new_caption = copy.deepcopy(self.new)

        # C) no caption or valid caption_id
        else:
            print("Provide either a new caption or valid caption id")

        # preprocess caption: clean and transform to either w2v or bow
        new_caption = self.preprocess_caption(caption=new_caption)

        # check if format is sparse
        if isspmatrix(new_caption):
            new_caption = new_caption.todense()

        # check if model is loaded
        if self.caption_model is None:
            if self.mode == "triplet_loss":
                try:
                    self.load_models_triplet_loss()
                except ValueError:
                    print("Model could not be loaded")
            else:
                try:
                    # TODO add model part 2
                    self.load_cross_modal_retrieval()
                except ValueError:
                    print("Model could not be loaded")

        self.new_embedding = self.caption_model(new_caption)


    def embed_new_image(self, image_id="361092202.jpg"):

        # load database_images in case a image_id is provided
        if self.new is None and image_id is not None:
                self.new_id = image_id
                try:
                    database_images = self.load_database_images()
                except ValueError:
                    print("database images could not be loaded")

                idx = np.where(database_images["id"] == image_id)[0][0]
                new_image_vector = database_images["x"][idx]
                # reshape to predict: has to be (1, F) format with F the dimensons of the embedding
                new_image_vector = new_image_vector.reshape(1, new_image_vector.shape[0])
                self.new = new_image_vector

        # check if there is a image vector
        elif self.new is not None:
            new_image_vector = self.new

        # no image vector or image_id
        else:
            print("provide either a new image vector or valid image_id")

        # check if model is loaded
        if self.image_model is None:
            if self.mode == "triplet_loss":
                try:
                    self.load_models_triplet_loss()
                except ValueError:
                    print("model could not be loaded")
            else:
                try:
                # TODO add model part 2
                    self.load_cross_modal_retrieval()
                except ValueError:
                    print("model could not be loaded")

        # embedding
        self.new_embedding = self.image_model.predict(new_image_vector)


    def preprocess_caption(self, caption, verbose=False):

        # clean caption
        if self.clean:
            _ = preprocessing.clean_descriptions(
                descriptions=caption,
                min_word_length=self.min_word_length,
                stem=self.stem,
                verbose=verbose,
                unique_only=self.unique_only
            )
        # convert caption to either bow or word2vec
        if self.caption_transformer is None:
            if self.mode == "triplet_loss":
                self.load_caption_transformer_triplet_loss()
            else:
                self.load_caption_transformer_cross_modal()
        trans = self.caption_transformer.transform(caption.values())
        return trans


    def rank(self, distance_metric="L2"):

        assert distance_metric in ["L2", "Hamming"]
        self.distance_metric=distance_metric

        if distance_metric == "L2":
            dist = norm(self.database["embedding"] - self.new_embedding, ord=2, axis=1)
        else:
            dist = 1 - np.mean((self.database["embedding"] - self.new_embedding == 0), axis=1)

        # find lowest indices
        lowest_idx = np.argsort(dist)[0:self.k]
        # get lowest distance
        lowest_dist = dist[lowest_idx].flatten().tolist()
        # get lowest ids
        lowest_ids = self.database["id"][lowest_idx].flatten().tolist()
        # return in dictionary format
        self.ranking = dict(zip(lowest_ids, lowest_dist))

    def prepare_image_database(self,
                               path_raw_data='include/input/image_features.csv',
                               image_model=None,
                               save_dir_database=None,
                               filename_database="database_images.pkl",
                               batch_size=512,
                               verbose=True):


        if image_model is None and self.image_model is None:
            if self.mode == "triplet_loss":
                self.load_models_triplet_loss()
            else:
                # TODO check if this works
                self.load_cross_modal_retrieval()
        elif image_model is not None:
            self.image_model = image_model
        else:
            raise Exception("Cannot find image model")

        if verbose:
            print("loading image features")
        database_images = pd.read_csv(path_raw_data, sep=" ", header=None)
        image_ids = database_images.iloc[:, 0].values
        images_x = database_images.iloc[:, 1:].values

        # loop to make embedding in batch sizes
        if verbose:
            print("embedding image features for database")
            # to print progress
            i = 0
            # needs to be greater than 1
            n = max(1, len(images_x) // batch_size)
        embedding = []
        for batch in get_batch(range(0, len(images_x)), batch_size):
            batch_embedding = self.image_model.predict(images_x[batch])
            embedding.append(batch_embedding)
            if verbose:
                i += 1
                print_progress_bar(i=i, maximum=n, post_text="Finish", n_bar=20)
        if verbose:
            print("\n")
        embedding = np.vstack(embedding)

        database_images = {
            "id": image_ids,
            "x": images_x,
            "embedding": embedding
        }
        # save
        if save_dir_database is not None:
            print("saving image database")
            if not os.path.exists(save_dir_database):
                os.makedirs(save_dir_database)
            f = open(os.path.join(save_dir_database, filename_database), "wb")
            pickle.dump(database_images, f)
        return database_images

    def prepare_caption_database(
            self,
            path_raw_data="include/input/results_20130124.token",
            path_json_format="include/output/data/",
            caption_model=None,
            batch_size=1024,
            save_dir_database=None,
            filename_database="database_captions.pkl",
            verbose=True):

        if caption_model is None and self.caption_model is None:
            if self.mode == "triplet_loss":
                self.load_models_triplet_loss()
            else:
                # TODO check if this works
                self.load_cross_modal_retrieval()
        elif caption_model is not None:
            self.caption_model = caption_model
        else:
            raise Exception("Cannot find image model")

        if verbose:
            print("loading captions")
        orig_captions = preprocessing.load_captions(path_raw_data=path_raw_data, dir_to_read_save=path_json_format)
        captions_x = copy.deepcopy(orig_captions)

        # TODO maybe also loop over preprocess_caption in batch sizes
        if verbose:
            print("preprocessing captions")
        captions_x = self.preprocess_caption(caption=captions_x)

        # loop to make embedding in batch sizes
        if verbose:
            print("embedding captions for database")
            # to print progress
            i = 0
            # needs to be greater than 1
            n = max(1, captions_x.shape[0] // batch_size)
        embedding = []
        for batch in get_batch(range(0, captions_x.shape[0]), batch_size):
            batch_X = captions_x[batch]
            if isspmatrix(batch_X):
                batch_X = batch_X.todense()
            batch_embedding = self.caption_model.predict(batch_X)
            embedding.append(batch_embedding)
            if verbose:
                i += 1
                print_progress_bar(i=i, maximum=n, post_text="Finish", n_bar=20)
        if verbose:
            print("\n")
        embedding = np.vstack(embedding)

        database_captions = {
            "original_captions": orig_captions,
            "id": np.array(list(orig_captions.keys())),
            "x": captions_x,
            "embedding": embedding
        }

        # save
        if save_dir_database is not None:
            print("saving caption database")
            if not os.path.exists(save_dir_database):
                os.makedirs(save_dir_database)
            joblib.dump(database_captions, os.path.join(save_dir_database, filename_database))
        return database_captions

    def plot_images(
            self,
            image_dir=None,
            nrows=2,
            ncols=5,
            figsize=(20, 10),
            title_fontsize=30,
            title_x=0.5,
            title_y=1.05):

        if image_dir is not None:
            self.image_dir = image_dir
        self.nrows = nrows
        self.ncols = ncols
        self.figsize = figsize
        self.title_fontsize = title_fontsize
        self.title_x = title_x
        self.title_y = title_y
        # TODO FIX UPPER TITLE
        # check if nrows and ncols matches ranking
        if self.k != nrows*ncols:
            nrows = ceil(self.k/ncols)

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        i = 0
        for key, value in self.ranking.items():
            img = cv2.imread(self.image_dir + key)
            axes.flatten()[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axes.flatten()[i].set(title=f'Distance: {round(value, 4)}')
            plt.setp(axes.flatten()[i].get_xticklabels(), visible=False)
            plt.setp(axes.flatten()[i].get_yticklabels(), visible=False)
            axes.flatten()[i].tick_params(axis='both', which='both', length=0)
            axes.flatten()[i].xaxis.grid(False)
            axes.flatten()[i].yaxis.grid(False)
            i += 1
        title = f"Ranking top {self.k} images"
        plt.suptitle(title, x=title_x, y=title_y, fontsize=title_fontsize)
        plt.tight_layout()
        plt.show()

    def print_captions(self,
                       image_dir=None,
                       figsize=(10, 8),
                       title_fontsize=20,
                       title_y=0.88,
                       title_x=0.5):

        # TODO FIX UPPER TITLE
        if image_dir is not None:
            self.image_dir = image_dir
        self.figsize=figsize
        self.title_fontsize=title_fontsize
        self.title_y=title_y
        self.title_x=title_x


        if self.new_id is not None:
            plt.figure(figsize=figsize)
            plt.grid(b=None)  # remove grid lines
            plt.axis('off')  # remove ticks axes
            img = cv2.imread(self.image_dir + self.new_id)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            # add suptitle with image id
            if self.new_id is not None:
                plt.suptitle(f" New image id: {self.new_id}", x=self.title_x, y=self.title_y, fontsize=self.title_fontsize)
            plt.show()

        captions_ids = list(self.ranking.keys())
        caption_distances = list(self.ranking.values())
        if self.database is None:
            self.database = self.load_database_captions()
        caption_text = []
        for i, id in enumerate(captions_ids):
            text = self.database['original_captions'][id]
            caption_text.append(text)
            print(f"Caption ID: {id}, Caption: {text}, Distance: {round(caption_distances[i], 4)}) ")
        # pandas data frame format
        out = [captions_ids, caption_text, caption_distances]
        # update format ranking (pandas dataframe instead of dictionary and also add original aption_text)
        return pd.DataFrame(out, index=["caption_id", "caption_text", "distance"]).T

    def new_image_pipeline(self, k=None, new=None, new_id=None):

        # number of images to rank
        if k is not None:
            self.k = k
        # add new caption
        if new is not None:
            self.new = new
        # add new image_id
        if new_id is not None:
            self.new_id = new_id

        # step 1) embed new image
        self.embed_new_image()
        # step 2) load caption database (already embedded)
        self.database = self.load_database_captions()
        # step 3) compute distance and rank
        self.rank()
        # step 4) visualize results
        self.print_captions()

    def new_caption_pipeline(self, k=None, new=None, new_id=None):

        # number of captions to rank
        if k is not None:
            self.k = k
        # add new image
        if new is not None:
            self.new = new
        # add new caption_id
        if new_id is not None:
            self.new_id = new_id

        # step 1) embed new caption
        self.embed_new_caption()
        # step 2) load image database (already embedded)
        self.database = self.load_database_images()
        # step 3) compute distance and rank
        self.rank()
        # step 4) print visualize results
        self.plot_images()