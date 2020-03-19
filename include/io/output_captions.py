# %% apply the one_hot encoding to has each caption
import csv

from include.bow import one_hot


def output_captions(captions, tokens, file_name="include/data/caption_features.csv", compress=True, verbose=True,
                    sep=''):
    """

    :param captions:
    :param tokens:
    :param file_name:
    :param compress:
    :param verbose:
    :param sep:
    :return:
    """

    with open(file_name, 'w', newline=sep) as f:
        writer = csv.writer(f)
        for i in range(len(captions)):
            out = one_hot.convert_to_bow(captions[i], tokens)
            if compress:
                captions[i].features = out.nonzero()[0]
            else:
                captions[i].features = out

            row = list([captions[i].image_id]) + list(captions[i].caption_id) + captions[i].features.tolist()
            writer.writerow(row)
            if i % 10000 == 0 and verbose:
                print(i)
