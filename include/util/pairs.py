import random


def make_dict(images, captions):
    data_dict = {}

    image_dict = {}

    for image in images:
        image_dict[image.image_name] = image

    for image in images:
        data_dict[image] = []

    for caption in captions:
        list = data_dict.get(image_dict[caption.image_id])
        list.append(caption)
        data_dict[image_dict[caption.image_id]] = list

    return data_dict


def get_pairs_images(dictionary):
    pairs = []
    index = 0
    for key, value in dictionary.items():
        index += 1
        for positive_caption in value:

            # take random negative example

            negative_caption = None
            while negative_caption is None or negative_caption.features == []:
                negative_key = random.choice(list(dictionary.keys()))
                while negative_key == key or dictionary.get(negative_key) == []:
                    negative_key = random.choice(list(dictionary.keys()))

                negative_caption = random.choice(dictionary.get(negative_key))


            pairs.append((key, positive_caption, negative_caption))

    return pairs

