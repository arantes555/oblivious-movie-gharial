import os

def safe_remove(path):
    if os.path.isfile(path):
        os.remove(path)
