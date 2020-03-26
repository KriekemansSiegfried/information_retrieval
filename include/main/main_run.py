

import include.bow.one_hot as one_hot

image_file_directory = 'include/data/flickr30k-images/'
caption_string = 'Two young guys with shaggy hair look at their hands while hanging out in the yard'


def set_image_file_directory(file_path):
    global image_file_directory
    image_file_directory = file_path


def enter_caption(new_caption_string):
    global caption_string
    caption_string = new_caption_string


def caption_to_feature(caption=caption_string):
    one_hot.get_one_hot(caption, _) # <- use vectorizor instead





