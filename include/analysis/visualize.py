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

