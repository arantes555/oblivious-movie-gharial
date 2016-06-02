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


def main():
    initialize_logger('.')
    bank = DocumentBank.DocumentBank()

    reviews_dir = './reviews_dataset/'
    success = 0
    failed = 0
    partial_success = 0
    logging.info('Starting to parse reviews')
    for file_name in os.listdir(reviews_dir)[:50]:
        with open(os.path.join(reviews_dir, file_name), encoding='latin-1') as file:
            try:
                doc = ReviewParser.parse(file.read())
                bank.add_document(doc['review'], {
                    'rating': doc['rating'],
                    'title': doc['movie'],
                    'author': doc['reviewer'],
                    'capsule_review': doc['capsule_review']
                })
                if doc['rating'] != '?':
                    success += 1
                else:
                    partial_success += 1
            except Exception as e:
                failed += 1
                logging.debug('Failed on %s : %s' % (file_name, str(e)))

    logging.info('Tried %i documents. Succeeded %i%% (%i%% of partial success). Failed: %i' %
                 (int(success + partial_success + failed),
                  int((success + partial_success) / (success + partial_success + failed) * 100),
                  int(partial_success / (success + partial_success) * 100),
                  int(failed)))
    bank.vectorize()
    bank.close()


if __name__ == '__main__':
    main()
