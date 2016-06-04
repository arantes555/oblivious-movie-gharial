import shelve
import logging
from sklearn.feature_extraction.text import CountVectorizer
from tinydb import TinyDB
from tinydb.storages import JSONStorage
from tinydb.middlewares import CachingMiddleware
import utils
import warnings
from numpy import argsort, zeros, array
import sklearn.feature_extraction.text as sktext
from sklearn.metrics import precision_score
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from time import time
from nltk.stem.snowball import EnglishStemmer
from nltk.tokenize import word_tokenize

with warnings.catch_warnings(record=True) as w:
    from nimfa import Snmf


def stem(text):
    stemmer = EnglishStemmer(ignore_stopwords=True)
    return " ".join([stemmer.stem(word) for word in word_tokenize(text)])


class Movie:
    def __init__(self, movie_id, reviews=None):
        self.id = movie_id
        self.reviews = []
        if reviews is not None:
            self.reviews.extend([{
                                     'userID': review['userID'],
                                     'rating': review['rating'],
                                     'review': stem(review['review'])
                                 } for review in reviews])

    def add_review(self, user_id, rating, review):
        self.reviews.append({
            'userID': user_id,
            'rating': rating,
            'review': stem(review)
        })

    def _generate_full_text(self):
        for review in self.reviews:
            yield review['review']

    def full_text(self):
        return ' '.join(self._generate_full_text())

    def serialize(self):
        return {
            'id': self.id,
            'reviews': self.reviews
        }

    def for_db(self):
        return {'content': self.full_text(), 'metadata': self.id, 'topic_ids': []}


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
            self.shelf['topic_names'] = {
                -1: 'No Topic'
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

    def add_document(self, content, metadata):
        """
        Add a single document to the bank
        :param content: content of the document
        :type content: str
        :param metadata: metadata around document
        :type metadata: dict
        """
        logging.debug('Inserting single document to tinydb')
        self.tinydb.insert({
            'metadata': metadata,
            'content': content,
            'topic_ids': []
        })

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
        self.shelf['vectorizer'] = CountVectorizer(decode_error='ignore',
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
            for document in sorted(self.tinydb.all(), key=lambda doc: doc.eid):
                yield document['content']

        features_matrix = self.shelf['vectorizer'].fit_transform(corpus())

        self.shelf['features_matrix'] = features_matrix

        # Inverse the vectorized vocabulary
        self.shelf['dictionnary'] = self.shelf['vectorizer'].get_feature_names()
        logging.info('Vectorizing done')
        self.shelf.sync()

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
            'beta': 1e+2
        }
        if options is not None:
            default_options.update(options)
        options = default_options

        logging.info('Starting topic extraction')
        # Basic feature selection
        keepwords = min(1000, self.shelf['features_matrix'].shape[1])
        logging.info('TF-IDF transforming the features matrix')
        m_tfidf = sktext.TfidfTransformer().fit_transform(self.shelf['features_matrix'])
        wrdtf = sum(m_tfidf).toarray()
        idx = argsort(wrdtf)[0][-keepwords:]
        data_ms = self.shelf['features_matrix'].tocsc()[:, idx]

        logging.info('Running the matrix factorization...')
        t0 = time()
        fctr = Snmf(data_ms, **options)
        fctr_res = fctr()
        logging.info('Matrix factorization done in %is' % int(time() - t0))

        rbas = fctr.coef().toarray()
        bas = zeros((rbas.shape[0], self.shelf['features_matrix'].shape[1]))
        bas[:, idx] = rbas

        (H, W) = bas, fctr_res.basis().toarray()

        # Calculate a best topic_id for each document, if too poor, associate -1
        yv = zeros(W.shape[0])
        mml = W.mean()
        for i in range(W.shape[0]):
            yv[i] = W[i].argmax()
            if W[i][int(yv[i])] < mml:
                yv[i] = -1  # Assign topic_id -1 to poorly categorized mails

        # Display and store topics
        for topic_idx, topic in enumerate(H):
            topbuf = " ".join(
                [self.shelf['dictionnary'][i]
                 for i in topic.argsort()[:-n_words - 1:-1]]
            )
            topic_name = "Topic #%i: %s" % (topic_idx, topbuf)
            self.shelf['topic_names'][int(topic_idx)] = topic_name
            logging.info(topic_name)

        logging.info('Updating database...')
        for document in sorted(self.tinydb.all(), key=lambda doc: doc.eid):
            self.tinydb.update({'topic_ids': [int(yv[document.eid - 1])]}, eids=[document.eid])
        return H, W  # Return (H,W) matrix factors

    def train_classifiers_fullset(self):
        """
        Trains the classifiers on the generated topics
        """
        logging.info('Start training classifiers')
        # Get list of labels from email versus label dict
        topic_ids = dict([(topic_id, [])
                          for document in sorted(self.tinydb.all(), key=lambda doc: doc.eid)
                          for topic_id in document['topic_ids']])
        logging.info('Dict of labels computed with size %i' % len(list(topic_ids.keys())))
        # Compute classifier for each label (except -1, which is no label)
        self.shelf['classifiers'] = {}
        for topic_id in topic_ids.keys():
            logging.info('Working on topic %s' % str(topic_id))
            if not topic_id == -1.0:  # TODO : Separate in multiple functions ...
                yvc = self._generate_yvc(topic_id)
                logging.info('generated YVC')
                # Check if enough positives
                length = len(yvc)
                total = sum(yvc)
                if total > -length + 20:  # TODO : "-len(yvc) + 20" ? Oo
                    logging.info('Respects strange condition')
                    # Launch classifier on feature matrix
                    ratio = (length - total) / (length + total)
                    self._classify(topic_id, yvc, ratio)

    def _generate_yvc(self, topic_id):
        """
        Helper method for "train_classifiers_fullset". Should not be used directly.
        :param topic_id: id of the topic
        :type topic_id: int
        """
        yvc = []
        # Scan each message in msg_id list (ordered as in DataM)
        for document in sorted(self.tinydb.all(), key=lambda doc: doc.eid):
            if topic_id in document['topic_ids']:
                yvc.append(1)
            else:
                yvc.append(-1)
        return yvc

    def _classify(self, topic_id, yvc, ratio):
        """
        Helper method for "train_classifiers_fullset". Should not be used directly.
        :param topic_id: ID of the current topic.
        :type topic_id: int
        :param yvc: Vector to classify.
        :type yvc: list
        :param ratio: Ratio of the yvc.
        :type ratio: float
        """
        logging.debug('Starting GridSearch on topic id %s' % topic_id)
        param_grid = [{'C': [0.1, 1, 10, 100], 'kernel': ['linear']}]
        scores = [('precision', precision_score)]  # TODO: refine SVM scores
        clf = GridSearchCV(SVC(C=1, class_weight={1: ratio}), param_grid, cv=5)
        clf.fit(self.shelf['features_matrix'], array(yvc))
        # Store best classifier in dict
        self.shelf['classifiers'][topic_id] = clf.best_estimator_
        # Report results
        logging.debug('GridSearch done')
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
