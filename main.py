import logging
from DocumentBank import DocumentBank
from parseReview import ReviewParser
from time import time
import os
import logger
import config


def main():
    logger.initialize('.')
    bank = DocumentBank()

    success = 0
    failed = 0
    partial_success = 0
    logging.info('Starting to parse reviews')
    t0 = time()
    files_names = os.listdir(config.REVIEWS_DIR)[:config.MAX_REVIEWS]
    files_number = len(files_names)
    progress = 0
    for i, file_name in enumerate(files_names):
        new_progress = int(i / files_number * 100)
        if new_progress != progress and new_progress % 10 == 0:
            logging.info('Progress: %i%%' % new_progress)
        progress = new_progress
        with open(os.path.join(config.REVIEWS_DIR, file_name), encoding='latin-1') as file:
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

    logging.info('Tried %i documents, %i%% full success, %i failed, in %is.' %
                 (int(files_number),
                  int(success / files_number * 100),
                  int(failed),
                  int(time() - t0)))
    bank.vectorize()
    bank.close()


if __name__ == '__main__':
    main()
