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

    reviews_dir = './reviews_dataset/'
    success = 0
    failed = 0
    partial_success = 0
    logging.info('Starting to parse reviews')
    t0 = time()
    for file_name in os.listdir(reviews_dir)[:config.MAX_REVIEWS]:
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

    logging.info('Tried %i documents, %i%% of partial success, %i failed, in %is.' %
                 (int(success + partial_success + failed),
                  int(partial_success / (success + partial_success) * 100),
                  int(failed),
                  int(time() - t0)))
    bank.vectorize()
    bank.close()


if __name__ == '__main__':
    main()
