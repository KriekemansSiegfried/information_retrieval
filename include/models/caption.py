class Caption:
    """ Class representing a caption in the dataset """

    def __init__(self, image_id, caption_id, tokens):
        self.image_id = image_id
        self.caption_id = caption_id
        self.tokens = tokens
