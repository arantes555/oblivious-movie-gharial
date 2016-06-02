import logging
import DocumentBank
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


def main():
    initialize_logger('.')
    bank = DocumentBank.DocumentBank('./test.db')
    bank.add_documents([{
        'content': 'test lzrkg ezrlkgerlg erlkgnerlkgn'
    }])
    bank.vectorize()
    bank.close()

if __name__ == '__main__':
    main()
