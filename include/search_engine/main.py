# ----------------------------------------------------------------
# 2) LOAD LIBRARIES
# ----------------------------------------------------------------
from include.search_engine.load_engine import search_engine
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
# interactive plotting: %matplotlib qt5

# TODO:
# - fix suptitle plot_images: should print the caption in the title
# - add functionality for part 2
# - make readme


# ----------------------------------------------------------------
# 1) GLOBAL VARIABLES
# ----------------------------------------------------------------

# %% GLOBAL VARIABLES (indicated in CAPITAL letters)

# COMMON PATHS

# path raw image features
PATH_RAW_IMAGE_FEATURES = 'include/input/image_features.csv'
# path raw caption features
PATH_RAW_CAPTION_FEATURES = "include/input/results_20130124.token"

# A) TRIPLET LOSS MODEL (TL) (PART 1)

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
# B) CROSS MODAL MODEl (PART 2)


# TODO add part 2

# %%
# ----------------------------------------------------------------
# 3) CREATE MODEL
# ----------------------------------------------------------------

# A) TRIPLET LOSS MODEL (TL) (PART 1)


# %% A.1) search engine for new caption

# Step 1: create search engine object
Se_new_caption = search_engine.SearchEngine(
    mode="triplet_loss",
    path_transformer=PATH_TRANSFORMER_TL,
    model_path=MODEL_PATH_TL,
    weights_path=WEIGHT_PATH_TL,
    database_images_path=DATABASE_IMAGE_DIR_TL + DATABASE_IMAGE_FILE_TL,
    database_captions_path=DATABASE_CAPTION_DIR_TL + DATABASE_CAPTION_FILE_TL
)
# %%
# Step 2: load database (you only need to do this once, except if you have a new model)
# This will take a moment (1min)
_ = Se_new_caption.prepare_image_database(
    path_raw_data=PATH_RAW_IMAGE_FEATURES,
    save_dir_database=DATABASE_IMAGE_DIR_TL,
    filename_database=DATABASE_IMAGE_FILE_TL,
    batch_size=512,
    verbose=True
)
# %%
# Step 3: run pipeline
# show 10 closest images for caption '361092202.jpg#4'
Se_new_caption.new_caption_pipeline(new_id='361092202.jpg#4', k=10)
plt.savefig("include/output/figures/triplet_loss/fig1_readme.png")
# %%
# add new caption and show 20 best photos
Se_new_caption.new_caption_pipeline(new="Water sea swimming", k=20)
plt.savefig("include/output/figures/triplet_loss/fig2_readme.png")
# %% A.2) search engine for new image

# Step 1: create search engine object
Se_new_image = search_engine.SearchEngine(
    mode="triplet_loss",
    path_transformer=PATH_TRANSFORMER_TL,
    model_path=MODEL_PATH_TL,
    weights_path=WEIGHT_PATH_TL,
    database_images_path=DATABASE_IMAGE_DIR_TL + DATABASE_IMAGE_FILE_TL,
    database_captions_path=DATABASE_CAPTION_DIR_TL + DATABASE_CAPTION_FILE_TL
)
# %%
# Step 2: load database (you only need to do this once, except if you have a new model)
# This will take a moment (1min)
_ = Se_new_image.prepare_caption_database(
    path_raw_data=PATH_RAW_CAPTION_FEATURES,
    save_dir_database=DATABASE_CAPTION_DIR_TL,
    filename_database=DATABASE_CAPTION_FILE_TL,
    batch_size=1024,
    verbose=True
)

# %%
# Step 3: run pipeline
# print top 10 captions for image "361092202.jpg"
Se_new_image.new_image_pipeline(new_id="361092202.jpg", k=10)



# B) CROSS MODAL MODEl (PART 2)

# TODO add part 2