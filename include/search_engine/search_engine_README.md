# Search engine
 - retrieves most relevant captions given a new image
 - retrieves most relevant images given a new caption or caption_id
 
 # Usage


## Import search engine 
```python
from include.search_engine.load_engine import search_engine
 
```
## Define Global Paths
```python
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
# TODO

```

## A Triplet loss modal (Part 1)
### A.1 given a new caption retrieve most similar images
#### Step 1:  create search engine object for a *new caption*

```python
Se_new_caption = search_engine.SearchEngine(
    mode="triplet_loss",
    path_transformer=PATH_TRANSFORMER_TL,
    model_path=MODEL_PATH_TL,
    weights_path=WEIGHT_PATH_TL,
    database_images_path=DATABASE_IMAGE_DIR_TL + DATABASE_IMAGE_FILE_TL,
    database_captions_path=DATABASE_CAPTION_DIR_TL + DATABASE_CAPTION_FILE_TL
)
```

#### Step 2: create *image database*
You only have to create this once! After you have ran this command, you can skip this block of code
This will take some time (around 1 min.) to run.
```python
_ = Se_new_caption.prepare_image_database(
    path_raw_data=PATH_RAW_IMAGE_FEATURES,
    save_dir_database=DATABASE_IMAGE_DIR_TL,
    filename_database=DATABASE_IMAGE_FILE_TL,
    batch_size=512,
    verbose=True
)
```
#### Step 3: run *caption pipeline*
```python
# show 10 closest images for caption '361092202.jpg#4'
Se_new_caption.new_caption_pipeline(new_id='361092202.jpg#4', k=10)
```

<img src="include/output/figures/triplet_loss/fig1_readme.png" alt="drawing" width="200" height="100"/>

```python
# new caption and show 20 most relevant images
Se_new_caption.new_caption_pipeline(new="Water sea swimming", k=20)
```
<img src="include/output/figures/triplet_loss/fig2_readme.png" alt="drawing" width="200" height="100"/>

### A.2 given a new image retrieve most similar captions

#### Step 1:  create search engine object for a *new image*

```python
Se_new_image = search_engine.SearchEngine(
    mode="triplet_loss",
    path_transformer=PATH_TRANSFORMER_TL,
    model_path=MODEL_PATH_TL,
    weights_path=WEIGHT_PATH_TL,
    database_images_path=DATABASE_IMAGE_DIR_TL + DATABASE_IMAGE_FILE_TL,
    database_captions_path=DATABASE_CAPTION_DIR_TL + DATABASE_CAPTION_FILE_TL
)
```
#### Step 2: create *caption database*
You only have to create this once! After you have ran this command, you can skip this block of code
This will take some time (around 1 min.) to run.
```python
_ = Se_new_image.prepare_caption_database(
    path_raw_data=PATH_RAW_CAPTION_FEATURES,
    save_dir_database=DATABASE_CAPTION_DIR_TL,
    filename_database=DATABASE_CAPTION_FILE_TL,
    batch_size=1024,
    verbose=True
)

```
#### Step 3: run *image pipeline*
```python
# print top 10 captions for image "361092202.jpg"
Se_new_image.new_image_pipeline(new_id="361092202.jpg", k=10)
```


## A Cross modal (Part 2)

*TODO*
