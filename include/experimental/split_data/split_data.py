import sys


def print_progress_bar(i, maximum, post_text="Finish", n_bar=10):
    """

    :param i:
    :param maximum:
    :param post_text:
    :param n_bar: size of progress bar (default 10)
    :return:
    """
    j = i / maximum
    sys.stdout.write('\r')
    sys.stdout.write(f"[{'=' * int(n_bar * j):{n_bar}s}] {int(100 * j)}%  {post_text}")
    sys.stdout.flush()


def train_val_test_set_desc(dic, train_idx, val_idx, test_idx, verbose=True):
    """

    :param dic:
    :param train_idx:
    :param val_idx:
    :param test_idx:
    :param verbose:
    :return:

    """
    n = len(dic)
    counter = 0
    train_dic = {}
    val_dic = {}
    test_dic = {}
    for key, value in dic.items():

        if verbose:
            print_progress_bar(i=counter, max=n)
        if key in train_idx:
            train_dic[key] = value
        elif key in val_idx:
            val_dic[key] = value
        elif key in test_idx:
            test_dic[key] = value
        else:
            print("image id not in train, validation or test set")
        counter += 1

    if verbose:
        print(f'length training set:{len(train_dic)}')
        print(f'length training set:{len(val_dic)}')
        print(f'length training set:{len(test_idx)}')

    return train_dic, val_dic, test_dic
