
class Image:

    def __init__(self, row):

        self.image_name = row[0]
        self.features = []
        for i in range(1,len(row)):
            self.features.append(row[i])