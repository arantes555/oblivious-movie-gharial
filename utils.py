import os
import re


def safe_remove(path):
    if os.path.isfile(path):
        os.remove(path)


def stop_words(path):
    tmp = []
    if os.path.isfile(path):
        with open(path, 'r') as f:
            for line in f:
                if isinstance(line, bytes):
                    tmp.append(re.sub(r'\n', '', str(line, 'utf-8')))
                else:
                    tmp.append(re.sub(r'\n', '', line))
            f.close()
    return tmp
