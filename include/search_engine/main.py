# ----------------------------------------------------------------
# 1) LOAD LIBRARIES
# ----------------------------------------------------------------
from include.search_engine.load_engine import search_engine
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
# interactive plotting: %matplotlib qt5

# TODO:
# - fix suptitle plot_images: should print the caption in the title
# - add part 2 to readme


# ----------------------------------------------------------------
# 2) GLOBAL VARIABLES
# ----------------------------------------------------------------

# %% GLOBAL VARIABLES (indicated in CAPITAL letters)

# ***********************************************
# COMMON PATHS
# ***********************************************

# path raw image features
PATH_RAW_IMAGE_FEATURES = 'include/input/image_features.csv'
# path raw caption features
PATH_RAW_CAPTION_FEATURES = "include/input/results_20130124.token"
# directory containg the images
IMAGE_DIR = "include/input/flickr30k-images/"

# ***********************************************
# A) TRIPLET LOSS MODEL (TL) (PART 1)
# ***********************************************

# path of saved bow or w2v model
PATH_TRANSFORMER_TL = 'include/output/model/triplet_loss/caption_bow_model.pkl'
# path of trained model
MODEL_PATH_TL = 'include/output/model/triplet_loss/best_model.json'
# path of weigths of trained model
WEIGHT_PATH_TL = 'include/output/model/triplet_loss/best_model.h5'
# directory to save image database
DATABASE_IMAGE_DIR_TL = "include/output/data/triplet_loss/database_images/"
# filename of image database
DATABASE_IMAGE_FILE_TL = "database_images.dat"
# directory to save image captions
DATABASE_CAPTION_DIR_TL = "include/output/data/triplet_loss/database_captions/"
# filename of captions database
DATABASE_CAPTION_FILE_TL = "database_captions.dat"

# ***********************************************
# B) CROSS MODAL MODEl (PART 2)
# ***********************************************

# path of saved bow or w2v model
PATH_TRANSFORMER_H = 'include/output/model/hashing/caption_bow_model.pkl'
# path models (both caption and image)
MODEL_PATH_H = 'include/output/model/hashing/model_best.pth.tar'
# directory to save image database
DATABASE_IMAGE_DIR_H = "include/output/data/hashing/database_images/"
# filename of image database
DATABASE_IMAGE_FILE_H = "database_images.pkl"
# directory to save image captions
DATABASE_CAPTION_DIR_H = "include/output/data/hashing/database_captions/"
# filename of captions database
DATABASE_CAPTION_FILE_H = "database_captions.pkl"


# -----------------------------------------------------------------------------------------
# 3) CREATE MODEL
# -----------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------
# B) # A) TRIPLET LOSS MODEL (TL) (PART 1)
# ------------------------------------------------------------------------------------------


# %% A.1) search engine for new caption

# Step 1: create search engine object
Se_new_caption_tl = search_engine.SearchEngine(
    mode="triplet_loss",
    path_transformer=PATH_TRANSFORMER_TL,
    model_path=MODEL_PATH_TL,
    weights_path=WEIGHT_PATH_TL,
    database_images_path=DATABASE_IMAGE_DIR_TL + DATABASE_IMAGE_FILE_TL,
    database_captions_path=DATABASE_CAPTION_DIR_TL + DATABASE_CAPTION_FILE_TL,
    image_dir=IMAGE_DIR
)
# %%
# Step 2: load database (you only need to do this once, except if you have a new model)
# This will take a moment (1min)
Se_new_caption_tl.prepare_image_database(
    path_raw_data=PATH_RAW_IMAGE_FEATURES,
    save_dir_database=DATABASE_IMAGE_DIR_TL,
    filename_database=DATABASE_IMAGE_FILE_TL,
    database_mode="all_data",   # test, training, val or all_data"
    batch_size=512,
    verbose=True
)
# %%
# Step 3: run pipeline
# show 10 closest images for caption '361092202.jpg#4'
Se_new_caption_tl.new_caption_pipeline(new_id='361092202.jpg#4', k=10)
print(Se_new_caption_tl.new)
plt.tight_layout()
plt.savefig("include/output/figures/triplet_loss/fig1_readme.png")
# %%
# add new caption and show 20 best photos
Se_new_caption_tl.new_caption_pipeline(new="Water sea swimming", k=20)
plt.savefig("include/output/figures/triplet_loss/fig2_readme.png")
# %% A.2) search engine for new image

# Step 1: create search engine object
Se_new_image_tl = search_engine.SearchEngine(
    mode="triplet_loss",
    path_transformer=PATH_TRANSFORMER_TL,
    model_path=MODEL_PATH_TL,
    weights_path=WEIGHT_PATH_TL,
    database_images_path=DATABASE_IMAGE_DIR_TL + DATABASE_IMAGE_FILE_TL,
    database_captions_path=DATABASE_CAPTION_DIR_TL + DATABASE_CAPTION_FILE_TL,
    image_dir=IMAGE_DIR
)
# %%
# Step 2: load database (you only need to do this once, except if you have a new model)
# This will take a moment (1min)
Se_new_image_tl.prepare_caption_database(
    path_raw_data=PATH_RAW_CAPTION_FEATURES,
    save_dir_database=DATABASE_CAPTION_DIR_TL,
    filename_database=DATABASE_CAPTION_FILE_TL,
    database_mode="all_data",   # test, training, val or all_data"
    batch_size=1024,
    verbose=True
)

# %%
# Step 3: run pipeline
# print top 10 captions for image "361092202.jpg"
Se_new_image_tl.new_image_pipeline(new_id="361092202.jpg", k=10)
plt.tight_layout()
plt.savefig("include/output/figures/triplet_loss/fig3_readme.png")

# ------------------------------------------------------------------------------------------
# B) HASHING (PART 2)
# ------------------------------------------------------------------------------------------
# %% B.1) search engine for new caption

# Step 1: create search engine object
Se_new_caption_h = search_engine.SearchEngine(
    mode="hashing",
    path_transformer=PATH_TRANSFORMER_H,
    model_path=MODEL_PATH_H,
    database_images_path=DATABASE_IMAGE_DIR_H + DATABASE_IMAGE_FILE_H,
    database_captions_path=DATABASE_CAPTION_DIR_H + DATABASE_CAPTION_FILE_H,
    image_dir=IMAGE_DIR
)
# %%
# Step 2: load database (you only need to do this once, except if you have a new model)
# This will take a moment (1min)
Se_new_caption_h.prepare_caption_database(
    path_raw_data=PATH_RAW_CAPTION_FEATURES,
    save_dir_database=DATABASE_CAPTION_DIR_H,
    filename_database=DATABASE_CAPTION_FILE_H,
    database_mode="all_data",   # test, training, val or all_data"
    batch_size=512,
    verbose=True
)

# %% Step 3: run pipeline
# show 10 closest images for caption '361092202.jpg#4'
Se_new_caption_h.new_caption_pipeline(new_id='361092202.jpg#4', k=10)
print(Se_new_caption_h.new)
plt.tight_layout()
plt.savefig("include/output/figures/hashing/fig1_readme.png")
# %%
# add new caption and show 20 best photos
Se_new_caption_h.new_caption_pipeline(new="Water sea swimming", k=20)
plt.savefig("include/output/figures/hashing/fig2_readme.png")


# %% B.2) search engine for new image

# Step 1: create search engine object
Se_new_image_h = search_engine.SearchEngine(
    mode="hashing",
    path_transformer=PATH_TRANSFORMER_H,
    model_path=MODEL_PATH_H,
    database_images_path=DATABASE_IMAGE_DIR_H + DATABASE_IMAGE_FILE_H,
    database_captions_path=DATABASE_CAPTION_DIR_H + DATABASE_CAPTION_FILE_H,
    image_dir=IMAGE_DIR
)
# %%
# Step 2: load database (you only need to do this once, except if you have a new model)
# This will take a moment (1min)
Se_new_image_h.prepare_caption_database(
    path_raw_data=PATH_RAW_CAPTION_FEATURES,
    save_dir_database=DATABASE_CAPTION_DIR_H,
    filename_database=DATABASE_CAPTION_FILE_H,
    database_mode="all_data",   # test, training, val or all_data"
    batch_size=1024,
    verbose=True
)

# %%
# Step 3: run pipeline
# print top 10 captions for image "361092202.jpg"
Se_new_image_h.new_image_pipeline(new_id="361092202.jpg", k=10)
plt.savefig("include/output/figures/hashing/fig3_readme.png")


# %%

while True:
    while True:
        model = input("model:")
        if model == "hash" or model == "triplet":
            break
        else:
            print("Please choose \"hash\" or \"triplet\"")
    while True:
        id_or_text = input("id or text:")
        if id_or_text == "id" or id_or_text == "text":
            break
        else:
            print("Please choose \"id\" or \"text\"")
    query = input("query:")

    if model == "triplet":
        if id_or_text == "id":
            Se_new_caption_tl.new_caption_pipeline(new_id=query, k=10)
            print(Se_new_caption_tl.new)
            plt.tight_layout()
            plt.show()
        else:
            Se_new_caption_tl.new_caption_pipeline(new=query, k=10)
            plt.tight_layout()
            plt.show()
    else:
        if id_or_text == "id":
            Se_new_caption_h.new_caption_pipeline(new_id=query, k=10)
            print(Se_new_caption_tl.new)
            plt.tight_layout()
            plt.show()

        else:
            Se_new_caption_h.new_caption_pipeline(new=query, k=10)
            plt.tight_layout()
            plt.show()