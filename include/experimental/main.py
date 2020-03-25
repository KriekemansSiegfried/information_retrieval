import re
import string


# %% load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r', encoding="utf8")
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


# extract descriptions for images
def load_descriptions(doc, first_description_only=True):
    mapping = dict()
    # process lines
    for line in doc.split('\n'):
        # split line by white space
        tokens = line.split()
        if len(line) < 2:
            continue
        # take the first token as the image id, the rest as the description
        image_id, image_desc = tokens[0], tokens[1:]
        # remove filename from image id
        image_id = image_id.split('.')[0]
        # convert description tokens back to string
        image_desc = ' '.join(image_desc)
        # store the first description for each image only
        if first_description_only:
            if image_id not in mapping:
                mapping[image_id] = image_desc

        # store all descriptions for each image
        else:
            if image_id not in mapping:
                mapping[image_id] = image_desc
            else:
                # add image description with a space added
                mapping[image_id] += image_desc.rjust(len(image_id) + 1)

    return mapping


# clean description text
def clean_descriptions(descriptions):
    # prepare regex for char filtering
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    for key, desc in descriptions.items():
        # tokenize
        desc = desc.split()
        # convert to lower case
        desc = [word.lower() for word in desc]
        # remove punctuation from each word
        desc = [re_punc.sub('', w) for w in desc]
        # remove hanging 's' and 'a'
        desc = [word for word in desc if len(word) > 1]
        # only store unique words
        unique_desc = []
        for word in desc:
            if word not in unique_desc:
                unique_desc.append(word)
        # store as string
        descriptions[key] = ' '.join(unique_desc)


# save descriptions to file, one per line
def save_doc(descriptions, filename):
    lines = list()
    for key, desc in descriptions.items():
        lines.append(key + ' ' + desc)
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()


# %%
filename = 'include/data/results_20130124.token'
# load descriptions
doc = load_doc(filename)
# parse descriptions (using all descriptions)
descriptions = load_descriptions(doc, first_description_only=False)
print('Loaded: %d ' % len(descriptions))
# %%
# clean descriptions
clean_descriptions(descriptions)
# summarize vocabulary (still additonal cleaning needed)
all_tokens = ' '.join(descriptions.values()).split()
vocabulary = set(all_tokens)
print('Vocabulary Size: %d' % len(vocabulary))
# save descriptions
save_doc(descriptions, 'include/data/descriptions.txt')
