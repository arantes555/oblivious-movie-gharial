import shelve
import logging
import utils
from tinydb import TinyDB
from tinydb.storages import JSONStorage
from tinydb.middlewares import CachingMiddleware

from numpy import argsort, zeros, array
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import precision_score
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from time import time
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
import config

import warnings
with warnings.catch_warnings(record=True) as w:
    from nimfa import Snmf

_stemmer = WordNetLemmatizer()
_tokenizer = RegexpTokenizer(r'\w+')


def tokenizer(text):
    return [_stemmer.lemmatize(word) for word in _tokenizer.tokenize(text)]


class Movie:
    def __init__(self, movie_id, title, reviews=None):
        self.id = movie_id
        self.title = title
        self.reviews = []
        if reviews is not None:
            self.reviews.extend(reviews)

    def add_review(self, user_id, rating, review):
        self.reviews.append({
            'userID': user_id,
            'rating': rating,
            'review': review
        })

    def _generate_full_text(self):
        for review in self.reviews:
            yield review['review']

    def full_text(self):
        return '\n'.join(self._generate_full_text())

    def serialize(self):
        return {
            'id': self.id,
            'title': self.title,
            'reviews': self.reviews
        }


class Topic:
    def __init__(self, _id, top_words):
        self.id = _id
        self.top_words = top_words

    def __str__(self):
        top_words = " ".join(self.top_words)
        return "Topic #%i: %s" % (self.id, top_words)

    def __repr__(self):
        if config.FULL_TOPICS:
            return self.__str__()
        else:
            return "#%i" % self.id


class DocumentBank:
    """
    Main class for storing documents and working on them
    """

    def __init__(self,
                 shelf_path='./bank.shelf',
                 tinydb_path='./documents.tinydb',
                 reset=True):
        """
        Initialize or load DocumentBank
        :param shelf_path: path to the shelf file storing a picklized dict
        :type shelf_path: str
        :param tinydb_path: path to the tinydb file storing the documents as a JSON
        :type tinydb_path: str
        :param reset: reset the document bank or load it, defaults to True
        :type reset: bool
        """
        if reset:
            logging.debug('DocumentBank is set to be reset')
            logging.debug('Removing shelf file if any')
            utils.safe_remove(shelf_path)
            logging.debug('Removing tinydb file if any')
            utils.safe_remove(tinydb_path)

        logging.debug('Loading (or creating) shelf')
        self.shelf = shelve.open(shelf_path, writeback=True)
        logging.debug('Loading (or creating) tinydb')
        self.tinydb = TinyDB(tinydb_path, storage=CachingMiddleware(JSONStorage))

        if reset:
            logging.debug('Initializing topic names')
            self.shelf['topics'] = {
                -1: Topic(-1, [])
            }
            logging.debug('Initializing vectorizer')
            self.shelf['vectorizer'] = None
            logging.debug('Initializing features matrix')
            self.shelf['features_matrix'] = None
            logging.debug('Initializing dictionnary')
            self.shelf['dictionnary'] = None
            logging.debug('Initializing classifiers')
            self.shelf['classifiers'] = None
            self.shelf.sync()

    def add_document(self, movie):
        """
        Add a single document to the bank
        :param movie: movie to add
        :type movie: Movie
        """
        logging.debug('Inserting single document to tinydb')
        self.tinydb.insert(movie.serialize())

    def add_documents(self, documents):
        """
        Add multiple documents to the bank
        :param documents: list of documents
        :type documents: list
        """
        logging.debug('Inserting multiple documents to tinydb')
        self.tinydb.insert_multiple(documents)

    def vectorize(self, stop_words=None, max_features=2000):
        """

        :param stop_words: list of stop words to give to the CountVectorizer
        :type stop_words: list
        :param max_features: maximum amount of features (words) to vectorize
        :type max_features: int
        """
        logging.info('Start vectorizing...')
        t0 = time()
        self.shelf['vectorizer'] = CountVectorizer(decode_error='ignore',
                                                   tokenizer=tokenizer,
                                                   strip_accents='unicode',
                                                   min_df=0.05,
                                                   max_df=0.80,
                                                   stop_words=stop_words,
                                                   max_features=max_features
                                                   )

        def corpus():
            """
            :return: Generator yielding all the documents content
            """
            for movie in sorted(self.tinydb.all(), key=lambda doc: doc.eid):
                yield Movie(movie['id'], movie['title'], movie['reviews']).full_text()

        features_matrix = self.shelf['vectorizer'].fit_transform(corpus())

        self.shelf['features_matrix'] = features_matrix

        # Inverse the vectorized vocabulary
        self.shelf['dictionnary'] = self.shelf['vectorizer'].get_feature_names()
        logging.info('Vectorizing done in %is' % int(time() - t0))
        self.shelf.sync()

    def _select_features(self):
        """
        Basic feature selection
        :return:
        """
        logging.debug('Selecting features...')
        t0 = time()
        logging.info('TF-IDF transforming the features matrix')
        m_tfidf = TfidfTransformer().fit_transform(self.shelf['features_matrix'])
        # TODO: dafuq is this sum ?
        wrdtf = sum(m_tfidf).toarray()
        # Keep 1000 words or less # TODO: dafuq ?
        keepwords = min(1000, self.shelf['features_matrix'].shape[1])
        # Get the `keepwords` first ids of the sorted wrdftf matrix
        # TODO: understand precisley what happens here with argsort
        idx = argsort(wrdtf)[0][-keepwords:]
        # TODO: Why transforming it into a CSC (compressed by column) instead keeping it a CSR (by rows)...
        # Only keeping the entries (rows or colums) corresponding to the kept words
        data_ms = self.shelf['features_matrix'].tocsc()[:, idx]
        logging.debug('Feature selection done in %is' % int(time() - t0))
        return data_ms, idx

    def _assign_topics(self, H, W, rank, n_words):
        logging.debug('Assigning topics to each entry...')
        t3 = time()
        # Calculate a best topic_id for each document, if too poor, associate -1
        yv = zeros(W.shape[0])
        mml = W.mean()
        counter = dict((i, []) for i in range(-1, rank))
        for i in range(W.shape[0]):
            movie = self.tinydb.get(eid=i + 1)
            yv[i] = W[i].argmax()
            if W[i][int(yv[i])] < mml:
                yv[i] = -1  # Assign topic_id -1 to poorly categorized documents
            counter[int(yv[i])].append({
                'id': movie['id'],
                'title': movie['title']
            })
        logging.debug('Topics were assigned in %is' % int(time() - t3))

        # Display and store topics
        for topic_idx, topic in enumerate(H):
            _topic = Topic(int(topic_idx), [self.shelf['dictionnary'][i]
                                            for i in topic.argsort()[:-n_words - 1:-1]])
            self.shelf['topics'][int(topic_idx)] = _topic
            logging.info('Topic #%i - %i movies: %s' % (_topic.id, len(counter[_topic.id]), " ".join(_topic.top_words)))
        logging.info('%i movie(s) were unassigned' % len(counter[-1]))

        logging.info('Updating database...')
        t4 = time()

        def assign_topic(element):
            element['topic_ids'] = [int(yv[element.eid - 1])]

        self.tinydb.update(assign_topic, eids=[i + 1 for i in range(len(self.tinydb))])
        self.shelf.sync()
        logging.info('Database updated in %is' % int(time() - t4))
        return counter

    def topic_extraction(self, options=None, n_words=12):
        """

        :param options: kwargs of nimfa.Snmf()
        :type options: dict
        :param n_words: amount of words to display per topic, doesn't impact qualty of topics
        :type n_words: int
        :return: H,W the two matrix factors
        :rtype: tuple
        """
        default_options = {
            # SNMF version to use ('l' or 'r')
            'version': 'r',
            # Number of topics to produce. This varies across samples, but should be kept relatively small.
            'rank': 20,
            # Seed are methods.seeding.nndsvd.Nndsvd Specify method to seed the computation of a factorization. If
            # specified :param:`W` and :param:`H` seeding must be None. If neither seeding method or initial fixed
            # factorization is specified, random initialization is used.
            'seed': 'random_c',
            # Maximum amount of iterations
            'max_iter': 12,
            # Part of the biclustering convergence test. It decides convergence if row clusters and column clusters have
            # not changed for i_conv convergence tests. It should have nonnegative value. Default value is 10.
            'i_conv': 10,
            # Part of the biclustering convergence test. It specifies the minimal allowance of the change of row
            # clusters. It should have nonnegative value.Default value is 0.
            'w_min_change': 0,
            # Used for suppressing Frobenius norm on the basis matrix (W).Default value is maximum value of the target
            # matrix (V).If eta is negative, maximum value of target matrix is used for it.
            'eta': 1.0,
            # It controls sparseness. Larger beta generates higher sparseness on H. Too large :param:`beta` is not
            # recommended. It should have positive value. Default value is 1e-4.
            'beta': 1e-4
        }
        if options is not None:
            default_options.update(options)
        options = default_options

        logging.info('Starting topic extraction...')
        data_ms, idx = self._select_features()

        logging.info('Running the matrix factorization...')
        t1 = time()
        fctr = Snmf(data_ms, **options)
        fctr_res = fctr()
        logging.info('Matrix factorization done in %is' % int(time() - t1))
        logging.debug('Extracting results...')
        t2 = time()
        # TODO: how the fuck does this give H and W ???
        # Matrix of mixture coefficients array
        rbas = fctr.coef().toarray()
        # Generate a matrix of 0
        bas = zeros((rbas.shape[0], self.shelf['features_matrix'].shape[1]))
        # Fill it with rbas it the right places
        bas[:, idx] = rbas
        (H, W) = bas, fctr_res.basis().toarray()
        logging.debug('Results extracted in %is' % int(time() - t2))
        return self._assign_topics(H, W, options['rank'], n_words)

    def train_classifiers_fullset(self, n_jobs=1, min_amount_relevant=10):
        """
        Trains the classifiers on the generated topics
        """
        min_amount_relevant = max(5, min_amount_relevant)
        logging.info('Start training classifiers with a minimum relevance of %i' % min_amount_relevant)
        # Compute classifier for each label (except -1, which is no label)
        topic_ids = (topic.id for topic in self.shelf['topics'].values() if topic.id != -1)
        self.shelf['classifiers'] = {}
        for topic_id in topic_ids:
            logging.debug('Working on topic #%i out of %i' % (topic_id, len(self.shelf['topics']) - 1))
            yvc = self._generate_yvc(topic_id)
            # Check if enough positives
            length = len(yvc)  # 2 * (amount of films with topic_id) - amount of films
            total = sum(yvc)  # amount of films
            # amount of films with topic_id > min_amount_relevant
            if ((total + length) / 2) > min_amount_relevant:
                t0 = time()
                logging.debug('Generating classifier for topic #%i' % topic_id)
                # Launch classifier on feature matrix
                self._classify(topic_id, yvc, (length - total) / (length + total), n_jobs)
                logging.info('Classifier #%i generated in %is' % (topic_id, int(time() - t0)))
            else:
                logging.info('Aborting classifier generation for topic #%i because it\'s not relevant' % topic_id)

    def _generate_yvc(self, topic_id):
        """
        Helper method for "train_classifiers_fullset". Should not be used directly.
        :param topic_id: id of the topic
        :type topic_id: int
        """
        yvc = []
        logging.debug('Generating YVC...')
        t0 = time()
        # Scan each message in msg_id list
        for document in sorted(self.tinydb.all(), key=lambda doc: doc.eid):
            if topic_id in document['topic_ids']:
                yvc.append(1)
            else:
                yvc.append(-1)
        logging.debug('YVC generated in %is' % int(time() - t0))
        return yvc

    def _classify(self, topic_id, yvc, ratio, n_jobs):
        """
        Helper method for "train_classifiers_fullset". Should not be used directly.
        :param topic_id: ID of the current topic.
        :type topic_id: int
        :param yvc: Vector to classify.
        :type yvc: list
        :param ratio: Ratio of the yvc.
        :type ratio: float
        """
        logging.debug('Starting GridSearch on topic id %i ...' % topic_id)
        t0 = time()
        param_grid = [{'C': [0.1, 1, 10, 100], 'kernel': ['linear']}]
        clf = GridSearchCV(SVC(C=1, class_weight={1: ratio}), param_grid, cv=5, n_jobs=n_jobs)
        clf.fit(self.shelf['features_matrix'], array(yvc))
        logging.debug('GridSearch on topic id %i done in %is' % (topic_id, int(time() - t0)))

        # Store best classifier in dict
        self.shelf['classifiers'][topic_id] = clf.best_estimator_
        # Report results
        scores = [('precision', precision_score)]  # TODO: refine SVM scores
        for score_name, score_func in scores:
            logging.debug("GridSearch on %s" % score_name)
            logging.debug("Best parameters set found on development set:")
            logging.debug(clf.best_estimator_)
            logging.debug("Grid scores on training set:")
            for params, mean_score, scores in clf.grid_scores_:
                logging.debug("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() / 2, params))

    def classify_document(self, content):
        """
        Assigns a topic to a given document
        :param content: document content
        :type content: str
        :return: list of topic ids
        :rtype: list
        """
        logging.debug('Classifying given document')
        # Tokenize body
        vecm = self.shelf['vectorizer'].transform([content]).toarray()

        # Produce topic list
        topic_ids = []
        for topic_id in self.shelf['classifiers']:
            # Use SVC model to classify mail for topic represented by topic_id
            resp = self.shelf['classifiers'][topic_id].predict(vecm)
            if resp > 0:
                topic_ids.append(topic_id)
        logging.debug('Classifying done')
        return topic_ids

    def close(self):
        """
        Method to close the bank, synchronizes the cache of the tinydb and of the shelf
        """
        logging.info("Closing bank")
        self.shelf.close()
        self.tinydb.close()
