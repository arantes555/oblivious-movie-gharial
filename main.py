import config
import logging
from DocumentBank import DocumentBank, Movie
from parseReview import HtmlReviewParser, AmazonReviewsParser
from time import time, strftime
import os
import logger
import utils
from random import shuffle
import json
from shutil import copyfile


def date():
    return strftime('%Y-%m-%d-%H:%M:%S')


def write_report(filename, report):
    with open('reports/' + filename + '.json', 'w') as outfile:
        json.dump(report, outfile, sort_keys=True, indent=2)


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
    # Initialization
    start_date = date()
    logger.initialize('.')
    bank = DocumentBank()

    # Fetching stopwords
    logging.info('Fetching stop words')
    stop_words = utils.stop_words(config.LANGUAGE_STOP_WORDS_PATH)
    stop_words.extend(utils.stop_words(config.PROJECT_STOP_WORDS_PATH))
    logging.info('Fetched %i stop words' % len(stop_words))

    n_movies = config.maxsize if config.READ_ALL_THEN_SHUFFLE else config.MOVIES_TO_CLASSIFY + config.MOVIES_TO_ANALYZE

    # Read reviews from disk
    n_reviews, movies_reviews = AmazonReviewsParser.from_json(config.AMAZON_REVIEWS_FILE,
                                                              meta=config.METADATA_FILE,
                                                              max_movies=n_movies)
    movies = [Movie(movie_id, movie['title'], [{
                                   'userID': review['reviewer_id'],
                                   'rating': review['score'],
                                   'review': review['review']
                               } for review in movie['reviews']])
              for movie_id, movie in movies_reviews.items()]

    # Shuffle the array, so that the movies to classify at the end aren't biased
    shuffle(movies)

    # Separate movies to add to the bank (and add them to it), and movies to classify afterwards
    movies_to_analyze = [movie for movie in movies[:config.MOVIES_TO_ANALYZE]]
    movies_to_classify = [movie for movie in movies[-config.MOVIES_TO_CLASSIFY:]]
    logging.info('Analyzing %i movies' % len(movies_to_analyze))
    bank.add_documents([movie.serialize() for movie in movies_to_analyze])

    # First vectorize the dataset
    bank.vectorize(stop_words=stop_words, max_features=config.MAX_FEATURES)

    # Then extract topics and assign them to movies in the dataset
    training_counter = bank.topic_extraction({'rank': config.N_TOPICS, 'beta': config.BETA}, n_words=config.N_TOP_WORDS)

    # Train the classifiers with the assigned topics
    bank.train_classifiers_fullset(n_jobs=config.N_JOBS,
                                   min_amount_relevant=int(config.MIN_RELEVANCE * len(movies_to_analyze)))

    # Retrieving results
    topics = bank.shelf['topics']
    classification_counter = dict((i, []) for i in range(-1, config.N_TOPICS))
    for movie in movies_to_classify:
        movie_topics = [topics[topic_id] for topic_id in
                        bank.classify_document(movie.full_text())]
        for topic in movie_topics:
            classification_counter[topic.id].append({
                'id': movie.id,
                'title': movie.title
            })
        if len(movie_topics):
            logging.info('Topics for document: %s: %s' % (movie.title, str(movie_topics)))
        else:
            classification_counter[-1].append(movie.title)
    for topic in classification_counter.keys():
        logging.info('Topic #%i: %i movies assigned' % (topic, len(classification_counter[topic])))
    logging.info('Managed to classify %i%% of the documents.' %
                 int((len(movies_to_classify) - len(classification_counter[-1])) / len(movies_to_classify) * 100))

    # Writing results to JSON
    report_filename = date()

    write_report(report_filename, {
        'start_date': start_date,
        'end_date': date(),
        'params': {
            'max_reviews': config.MAX_REVIEWS,
            'max_features': config.MAX_FEATURES,
            'min_relevance': config.MIN_RELEVANCE,
            'n_topics': config.N_TOPICS,
            'n_reviews': n_reviews,
            'n_movies': len(movies),
            'n_movies_training': len(movies_to_analyze),
            'n_movies_classify': len(movies_to_classify),
            'beta': config.BETA,
        },
        'results': [{
                        'topic': topics[topic_id].top_words,
                        'training_movies_in_topic': training_counter[topic_id],
                        'classification_movies_in_topic': classification_counter[topic_id]
                    } for topic_id in topics] + [{
            'topic': ['No Topic'],
            'training_movies_in_topic': training_counter[-1],
            'classification_movies_in_topic': classification_counter[-1]
        }]
    })
    bank.close()
    copyfile('./all.log', './reports/%s.log' % report_filename)


if __name__ == '__main__':
    main()
