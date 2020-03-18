from include.models.caption import Caption


def convert_to_caption(line):
    first_split = line.split("#", 1)
    second_split = first_split[1].split("\t", 1)

    image_id = first_split[0]
    caption_id = second_split[0].strip()
    tokens = second_split[1].split(" ")
    return Caption(image_id, caption_id, tokens)

def import_captions(filename):
    captions = []

    with open(filename, mode='r', encoding="utf8") as f:
        lines = f.readlines()
        for line in lines:
            captions.append(convert_to_caption(line))

    return captions
