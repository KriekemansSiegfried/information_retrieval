from numpy import float64


class Image:
    """ Class representing an image in the dataset """

    def __init__(self, row):

        self.image_name = row[0]
        self.features = []
        for i in range(1,len(row)):
            self.features.append(float64(row[i]))