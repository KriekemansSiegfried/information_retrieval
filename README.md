# Project Text based Information Retrieval

## Team members:

 - Giel Indekeu
 - Siegfried Kriekemans
 - Pieter-Jan Inghelbrecht

# Search engine
 - retrieves most relevant captions given a new image
 - retrieves most relevant images given a new caption or caption_id
 
 # Usage


## Import search engine 
```python
from include.search_engine.load_engine import search_engine
```
## Define Global Paths
### Common Paths
```python

# path raw image features
PATH_RAW_IMAGE_FEATURES = 'include/input/image_features.csv'
# path raw caption features
PATH_RAW_CAPTION_FEATURES = "include/input/results_20130124.token"
# directory containg the images
IMAGE_DIR = "include/input/flickr30k-images/"
```
### A) Triplet Loss Model (TL) (Part 1)
```python

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

```
### B) Hashing (Part 2)
```python

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

```

## A) Triplet loss modal (Part 1)
### A.1) given a new caption, retrieve most similar images
#### Step 1:  create search engine object for a *new caption*

```python
Se_new_caption_tl = search_engine.SearchEngine(
    mode="triplet_loss",
    path_transformer=PATH_TRANSFORMER_TL,
    model_path=MODEL_PATH_TL,
    weights_path=WEIGHT_PATH_TL,
    database_images_path=DATABASE_IMAGE_DIR_TL + DATABASE_IMAGE_FILE_TL,
    database_captions_path=DATABASE_CAPTION_DIR_TL + DATABASE_CAPTION_FILE_TL,
    image_dir=IMAGE_DIR
)
```

#### Step 2: create *image database*
You only have to create this once! After you have ran this block of code the *image database* will be stored locally.
This will take some time (around 1 min.) to run.
```python
Se_new_caption_tl.prepare_image_database(
    path_raw_data=PATH_RAW_IMAGE_FEATURES,
    save_dir_database=DATABASE_IMAGE_DIR_TL,
    filename_database=DATABASE_IMAGE_FILE_TL,
    database_mode="all_data",   # test, training, val or all_data"
    batch_size=512,
    verbose=True
)
```
#### Step 3: run *caption pipeline*
```python
# show 10 closest images for caption '361092202.jpg#4'
Se_new_caption_tl.new_caption_pipeline(new_id='361092202.jpg#4', k=10)
print(Se_new_caption_tl.new)
plt.tight_layout()
```

```python
>>> {'361092202.jpg#4': 'A hiker discovers a feature in an otherwise barren landscape .'}
```

<img src="https://github.com/KriekemansSiegfried/information_retrieval/blob/master/include/output/figures/triplet_loss/figure1_readme.png" />


```python
# new caption and show 20 most relevant images
Se_new_caption_tl.new_caption_pipeline(new="Group of children are playing", k=20)
```
<img src="https://github.com/KriekemansSiegfried/information_retrieval/blob/master/include/output/figures/triplet_loss/figure2_readme.png" />

### A.2) given a new image, retrieve most similar captions

#### Step 1:  create search engine object for a *new image*

```python
Se_new_image_tl = search_engine.SearchEngine(
    mode="triplet_loss",
    path_transformer=PATH_TRANSFORMER_TL,
    model_path=MODEL_PATH_TL,
    weights_path=WEIGHT_PATH_TL,
    database_images_path=DATABASE_IMAGE_DIR_TL + DATABASE_IMAGE_FILE_TL,
    database_captions_path=DATABASE_CAPTION_DIR_TL + DATABASE_CAPTION_FILE_TL,
    image_dir=IMAGE_DIR
)
```
#### Step 2: create *caption database*
You only have to create this once! After you have ran this block of code, the *caption database* will be stored locally.
This will take some time (around 1 min.) to run.
```python
Se_new_image_tl.prepare_caption_database(
    path_raw_data=PATH_RAW_CAPTION_FEATURES,
    save_dir_database=DATABASE_CAPTION_DIR_TL,
    filename_database=DATABASE_CAPTION_FILE_TL,
    database_mode="all_data",   # test, training, val or all_data"
    batch_size=1024,
    verbose=True
)
```
#### Step 3: run *image pipeline*
```python
# print top 10 captions for image "361092202.jpg"
Se_new_image_tl.new_image_pipeline(new_id="361092202.jpg", k=10)
plt.tight_layout()
```

<img src="https://github.com/KriekemansSiegfried/information_retrieval/blob/master/include/output/figures/triplet_loss/figure3_readme.png" />

Top 10 most relevant captions (ranked)
```python
>>> Caption ID: 445148321.jpg#0, Caption: A person in the distance hikes among hoodoos with stars visible in the sky ., Distance: 22.0601) 
>>> Caption ID: 3944884778.jpg#0, Caption: A line of hikers trek across the rocky , sandy soil toward the summit on a hazy day ., Distance: 22.0689) 
>>> Caption ID: 2245989501.jpg#1, Caption: A man leads a caravan of six camels and their riders up a sandy hill , with rocky mountains in the background ., Distance: 22.0715) 
>>> Caption ID: 2555545571.jpg#3, Caption: Person with afro shaking hands with crowd ., Distance: 22.0828) 
>>> Caption ID: 445148321.jpg#4, Caption: The night sky in the desert ., Distance: 22.1819) 
>>> Caption ID: 2544134113.jpg#0, Caption: The silhouettes of men against a cloudy yet bright sky ., Distance: 22.1868) 
>>> Caption ID: 2245989501.jpg#3, Caption: Several persons on the backs of camels traversing a hill ., Distance: 22.2413) 
>>> Caption ID: 425895906.jpg#4, Caption: A man and his camel in the dusty desert ., Distance: 22.3032) 
>>> Caption ID: 3224560800.jpg#2, Caption: A man dances with a fire baton at night ., Distance: 22.3338) 
>>> Caption ID: 352981175.jpg#0, Caption: One mountaineer is kneeling on the ground next to another mountaineer who is standing ., Distance: 22.3689) 

```

## B) Hashing (Part 2)
### B.1) given a new caption, retrieve most similar images
#### Step 1:  create search engine object for a *new caption*
```python
Se_new_caption_h = search_engine.SearchEngine(
    mode="hashing",
    path_transformer=PATH_TRANSFORMER_H,
    model_path=MODEL_PATH_H,
    database_images_path=DATABASE_IMAGE_DIR_H + DATABASE_IMAGE_FILE_H,
    database_captions_path=DATABASE_CAPTION_DIR_H + DATABASE_CAPTION_FILE_H,
    image_dir=IMAGE_DIR
)
```

#### Step 2: create *image database*
You only have to create this once! After you have ran this block of code the *image database* will be stored locally.
This will take some time (around 1 min.) to run.
```python
Se_new_caption_h.prepare_image_database(
    path_raw_data=PATH_RAW_IMAGE_FEATURES,
    save_dir_database=DATABASE_IMAGE_DIR_H,
    filename_database=DATABASE_IMAGE_FILE_H,
    database_mode="all_data",   # test, training, val or all_data"
    batch_size=512,
    verbose=True
)
```
#### Step 3: run *caption pipeline*
```python
# show 10 closest images for caption '361092202.jpg#4'
Se_new_caption_h.new_caption_pipeline(new_id='361092202.jpg#4', k=10)
print(Se_new_caption_h.new)
plt.tight_layout()
```

```python
>>> {'361092202.jpg#4': 'A hiker discovers a feature in an otherwise barren landscape .'}
```

<img src="https://github.com/KriekemansSiegfried/information_retrieval/blob/master/include/output/figures/hashing/fig1_readme.png" />

```python
# new caption and show 20 most relevant images
Se_new_caption_h.new_caption_pipeline(new="Water sea swimming", k=20)
```
<img src="https://github.com/KriekemansSiegfried/information_retrieval/blob/master/include/output/figures/hashing/fig2_readme.png" />

### B.2) given a new image, retrieve most similar captions

#### Step 1:  create search engine object for a *new image*

```python
Se_new_image_h = search_engine.SearchEngine(
    mode="hashing",
    path_transformer=PATH_TRANSFORMER_H,
    model_path=MODEL_PATH_H,
    database_images_path=DATABASE_IMAGE_DIR_H + DATABASE_IMAGE_FILE_H,
    database_captions_path=DATABASE_CAPTION_DIR_H + DATABASE_CAPTION_FILE_H,
    image_dir=IMAGE_DIR
)
```
#### Step 2: create *caption database*
You only have to create this once! After you have ran this block of code, the *caption database* will be stored locally.
This will take some time (around 1 min.) to run.
```python
Se_new_image_h.prepare_caption_database(
    path_raw_data=PATH_RAW_CAPTION_FEATURES,
    save_dir_database=DATABASE_CAPTION_DIR_H,
    filename_database=DATABASE_CAPTION_FILE_H,
    database_mode="all_data",   # test, training, val or all_data"
    batch_size=1024,
    verbose=True
)
```
#### Step 3: run *image pipeline*
```python
# print top 10 captions for image "361092202.jpg"
Se_new_image_h.new_image_pipeline(new_id="361092202.jpg", k=10)
plt.tight_layout()
```

<img src="https://github.com/KriekemansSiegfried/information_retrieval/blob/master/include/output/figures/hashing/fig3_readme.png" />

Top 10 most relevant captions (ranked)

```python
>>> Caption ID: 4689266358.jpg#0, Caption: A main street scene of a small town with an overhead welcome sign that says " Welcome to Golden " ., Distance: 0.1654) 
>>> Caption ID: 4896173039.jpg#0, Caption: A woman wearing a black sari with orange floral print reads a book with a purple cover ., Distance: 0.1671) 
>>> Caption ID: 7655480476.jpg#2, Caption: Group of tourist 's mostly women having fun on a bridge ., Distance: 0.1676) 
>>> Caption ID: 4271560578.jpg#1, Caption: A man on a bicycle , wearing an all black outfit , looks as though he 's hanging in midair over a bridge ., Distance: 0.1677) 
>>> Caption ID: 3747821314.jpg#0, Caption: A man wearing a black hat , a tan and black plaid suit and an orange shirt performs behind a man who is wearing an opened black shirt holds a piece of paper and sings into a microphone ., Distance: 0.1681) 
>>> Caption ID: 1045309098.jpg#0, Caption: Juggling her shopping bags and Betty Boop backpack and young woman crosses a city street ., Distance: 0.1682) 
>>> Caption ID: 6744211811.jpg#1, Caption: A small , skinny , black child hanging on a piece of wood looking at a white woman ., Distance: 0.1683) 
>>> Caption ID: 4166059793.jpg#1, Caption: A woman wearing blue clothing shows a picture book to several small children ., Distance: 0.1683) 
>>> Caption ID: 1001896054.jpg#1, Caption: John Deere tractors cruises down a street , while the driver wears easy to see clothing ., Distance: 0.1686) 
>>> Caption ID: 3483392163.jpg#2, Caption: Men in black clothing and women in white shirts some with color , reading or singing from a book ., Distance: 0.1689) 
```