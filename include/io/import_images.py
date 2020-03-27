import csv
import pandas as pd
from include.models.image import Image


def import_images(filename):
    """

    :param filename:
    :return:
    """
    images = []
    with open(filename, mode='r', encoding="utf8") as file:
        csv_reader = csv.reader(file, delimiter=' ')
        for line in csv_reader:
            if len(line) > 0:
                images.append(Image(line))
    return images


def import_images_as_ndarray(filename):
    df_image = pd.read_csv(filename, sep=" ", header=None)
    image_array = df_image.iloc[:, :].values
    return image_array
