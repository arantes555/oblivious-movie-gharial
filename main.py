import logging
import DocumentBank
from parseReview import ReviewParser
import os


def initialize_logger(output_dir):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # create console handler and set level to info
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # create error file handler and set level to error
    handler = logging.FileHandler(os.path.join(output_dir, "error.log"), "w", encoding=None, delay="true")
    handler.setLevel(logging.ERROR)
    formatter = logging.Formatter("%(asctime)s %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # create debug file handler and set level to debug
    handler = logging.FileHandler(os.path.join(output_dir, "all.log"), "w")
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def list_files(path):
    # returns a list of names (with extension, without full path) of all files
    # in folder path
    files = []
    for name in os.listdir(path):
        if os.path.isfile(os.path.join(path, name)):
            files.append(name)
    return files


def main():
    initialize_logger('.')
    bank = DocumentBank.DocumentBank('./test.db')

    files_list = list_files('./reviews_dataset')
    documents_tried = 0
    documents_success = 0

    for file_name in files_list:
        documents_tried += 1
        print('Trying : ', file_name)
        with open('./reviews_dataset/' + file_name) as file:
            try:
                data = file.read()
                doc = ReviewParser.parse(data)
                bank.add_documents([doc])
                documents_success += 1
            except:
                pass

    bank.vectorize()
    bank.close()


if __name__ == '__main__':
    main()
