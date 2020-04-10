import sys


def print_progress_bar(i=None, maximum=None, post_text="Finish", n_bar=10):
    """
        Progress bar of your operation

    :param i: Integer

    :param maximum: Integer

    :param post_text: String

    :param n_bar: Integer
        size of progress bar (default 10)
    :return:
        prints out a progress bar
    """
    j = i / maximum
    sys.stdout.write('\r')
    sys.stdout.write(f"[{'=' * int(n_bar * j):{n_bar}s}] {int(100 * j)}%  {post_text}")
    sys.stdout.flush()
