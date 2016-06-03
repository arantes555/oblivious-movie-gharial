import logging
from DocumentBank import DocumentBank
from parseReview import HtmlReviewParser, AmazonReviewsParser
from time import time
import os
import logger
import config
import utils


def append_html_reviews_to_bank(bank, reviews_path, max_reviews):
    logging.info('Starting to parse reviews')
    # Only for debugging and logging purposes
    t0 = time()
    success = 0
    failed = 0
    partial_success = 0
    progress = 0

    files_names = os.listdir(reviews_path)[:max_reviews]
    files_number = len(files_names)

    for i, file_name in enumerate(files_names):
        # Logging progress
        new_progress = int(i / files_number * 100)
        if new_progress != progress and new_progress % 10 == 0:
            logging.info('Progress: %i%%' % new_progress)
        progress = new_progress
        # do the work
        with open(os.path.join(reviews_path, file_name), encoding='latin-1') as file:
            try:
                doc = HtmlReviewParser.parse(file.read())
                bank.add_document(doc.pop('review'), doc)
                if doc['rating'] != '?':
                    success += 1
                else:
                    # Rating parsing often fails, so if it's a ? it's considered a partial success
                    partial_success += 1
            except Exception as e:
                failed += 1
                logging.debug('Failed on %s : %s' % (file_name, str(e)))

    logging.info('Tried %i documents, %i%% full success, %i failed, in %is.' %
                 (int(files_number),
                  int(success / files_number * 100),
                  int(failed),
                  int(time() - t0)))


def main():
    logger.initialize('.')
    bank = DocumentBank()

    logging.info('Fetching stop words')
    stop_words = utils.stop_words(config.LANGUAGE_STOP_WORDS_PATH)
    stop_words.extend(utils.stop_words(config.PROJECT_STOP_WORDS_PATH))
    logging.info('Fetched %i stop words' % len(stop_words))

    reviews = AmazonReviewsParser.from_file(config.AMAZON_REVIEWS_FILE,
                                            max_reviews=(config.MAX_DOCUMENTS_ANALYZE + config.DOCUMENTS_CLASSIFY))
    reviews = [{'content': doc.pop('review'), 'metadata': doc} for doc in reviews]
    reviews_to_classify = reviews[-config.DOCUMENTS_CLASSIFY:]
    reviews_to_analyze = reviews[:-config.DOCUMENTS_CLASSIFY]
    bank.add_documents(reviews_to_analyze)

    bank.vectorize(stop_words=stop_words, max_features=config.MAX_FEATURES)

    bank.topic_extraction()

    bank.train_classifiers_fullset()

    for doc in reviews_to_classify:
        print([bank.shelf['topic_names'][label] for label in bank.classify_document(doc['content'])])

    bank.close()


if __name__ == '__main__':
    main()
