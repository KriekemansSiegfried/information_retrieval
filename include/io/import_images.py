import csv

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
