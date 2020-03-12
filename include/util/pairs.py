import random


def make_dict(images, captions):
    data_dict = {}
    for image in images:
        data_dict[image.image_name] = []

    for caption in captions:
        list = data_dict.get(caption.image_id)
        list.append(caption)
        data_dict[caption.image_id] = list

    return data_dict


def get_pairs_images(dictionary):
    pairs = []
    for key, value in dictionary.items():
        for positive_caption in value:

            # take random negative example
            negative_key = random.choice(list(dictionary.keys()))
            while negative_key == key:
                negative_key = random.choice(list(dictionary.keys()))

            negative_caption = random.choice(dictionary.get(negative_key))

            pairs.append((key, positive_caption, negative_caption))
            break
        break

    print(pairs[0])
    return pairs
