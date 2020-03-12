import csv

from include.models.image import Image


def import_images(filename):
    images = []
    with open(filename) as file:
        csv_reader = csv.reader(file, delimiter=' ')
        for line in csv_reader:
            if len(line) > 0:
                images.append(Image(line))