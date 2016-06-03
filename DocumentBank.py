import shelve
import logging
from sklearn.feature_extraction.text import CountVectorizer
from tinydb import TinyDB, Query
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
import config

with warnings.catch_warnings(record=True) as w:
    from nimfa import Snmf


class DocumentBank:
    def __init__(self,
                 shelf_path='./bank.shelf',
                 tinydb_path='./documents.tinydb',
                 reset=True):
        if reset:
            utils.safe_remove(shelf_path)
            utils.safe_remove(tinydb_path)

        self.shelf = shelve.open(shelf_path, writeback=True)
        self.tinydb = TinyDB(tinydb_path, storage=CachingMiddleware(JSONStorage))
        self.shelf['topic_names'] = {
            -1: 'No Topic'
        }

    def add_document(self, document_content, document_metadata):
        labels = []

        self.tinydb.insert({
            'metadata': document_metadata,
            'content': document_content,
            'labels': labels
        })

    def add_documents(self, documents):
        self.tinydb.insert_multiple(documents)

    def vectorize(self, stop_words=None, max_features=2000):
        logging.info('Start vectorizing...')
        self.shelf['vectorizer'] = CountVectorizer(decode_error='ignore',
                                                   strip_accents='unicode',
                                                   min_df=0.05,
                                                   max_df=0.80,
                                                   stop_words=stop_words,
                                                   max_features=max_features
                                                   )

        def corpus():
            for document in sorted(self.tinydb.all(), key=lambda doc: doc.eid):
                yield document['content']

        features_matrix = self.shelf['vectorizer'].fit_transform(corpus())

        self.shelf['features_matrix'] = features_matrix

        # Inverse the vectorized vocabulary
        self.shelf['dictionnary'] = self.shelf['vectorizer'].get_feature_names()
        logging.info('Vectorizing done')
        self.shelf.sync()

    def topic_extraction(self, options=None, n_words=12):
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

        # Create artificial user folders/labels from NMF topic results
        # (the Melanion dataset does not have any labels)
        yv = zeros(W.shape[0])
        mml = W.mean()
        for i in range(W.shape[0]):
            yv[i] = W[i].argmax()
            if W[i][int(yv[i])] < mml:
                yv[i] = -1  # Assign label -1 to poorly categorized mails

        # Display topics
        for topic_idx, topic in enumerate(H):
            topbuf = " ".join(
                [self.shelf['dictionnary'][i]
                 for i in topic.argsort()[:-config.N_TOP_WORDS - 1:-1]]
            )
            topic_name = "Topic #%i: %s" % (topic_idx, topbuf)
            self.shelf['topic_names'][int(topic_idx)] = topic_name
            logging.info(topic_name)

        for document in sorted(self.tinydb.all(), key=lambda doc: doc.eid):
            self.tinydb.update({'labels': [int(yv[document.eid - 1])]}, eids=[document.eid])
        return H, W  # Return (H,W) matrix factors

    def train_classifiers_fullset(self):
        """
        Compute classifiers for full email set, overwrites existing classifiers.
        Uses list of all unique labels computed from self.mail_db.data['labels']
        """
        logging.info('Start training classifiers')
        # Get list of labels from email versus label dict
        labels = dict([(label, [])
                       for document in sorted(self.tinydb.all(), key=lambda doc: doc.eid)
                       for label in document['labels']])
        logging.info('Dict of labels computed with size %i' % len(list(labels.keys())))
        # Compute classifier for each label (except -1, which is no label)
        self.shelf['classifiers'] = {}
        for label in labels.keys():
            logging.info('Working on label %s' % str(label))
            if not label == -1.0:  # TODO : Separate in multiple functions ...
                yvc = self._generate_yvc(label)
                logging.info('generated YVC')
                # Check if enough positives
                length = len(yvc)
                total = sum(yvc)
                if total > -length + 20:  # TODO : "-len(yvc) + 20" ? Oo
                    logging.info('Respects strange condition')
                    # Launch classifier on feature matrix
                    ratio = (length - total) / (length + total)
                    self._classify(label, yvc, ratio)

    def _generate_yvc(self, label):
        """
        Helper method for "train_classifiers_fullset". Should not be used directly.
        Parameters
        ----------
        l_id: float
            ID of the Label for which to compute the yvc.
        """
        yvc = []
        # Scan each message in msg_id list (ordered as in DataM)
        for document in sorted(self.tinydb.all(), key=lambda doc: doc.eid):
            if label in document['labels']:
                yvc.append(1)
            else:
                yvc.append(-1)
        return yvc

    def _classify(self, l_id, yvc, ratio):
        """
        Helper method for "train_classifiers_fullset". Should not be used directly.
        Parameters
        ----------
        l_id: float
            ID of the current Label.
        yvc: list
            Vector to classify.
        ratio: float
            Ratio of the yvc.
        """
        param_grid = [{'C': [0.1, 1, 10, 100], 'kernel': ['linear']}]
        scores = [('precision', precision_score)]  # TODO: refine SVM scores
        clf = GridSearchCV(SVC(C=1, class_weight={1: ratio}), param_grid, cv=5)
        clf.fit(self.shelf['features_matrix'], array(yvc))
        # clf = GridSearchCV(SVC(C=1,class_weight={1:ratio}), param_grid)
        # clf.fit(self.mail_db.features_matrix, array(yvc), cv=5)
        # Store best classifier in dict
        self.shelf['classifiers'][l_id] = clf.best_estimator_
        # Report results
        for score_name, score_func in scores:
            print(19 * '=' + ' Grid search on ' + score_name + 19 * '=')
            print("Best parameters set found on development set:")
            print(clf.best_estimator_)
            print("Grid scores on training set:")
            for params, mean_score, scores in clf.grid_scores_:
                print("%0.3f (+/-%0.03f) for %r" % (
                    mean_score, scores.std() / 2, params))
        print('Done...')

    def classify_document(self, content):
        # Tokenize body
        vecm = self.shelf['vectorizer'].transform([content]).toarray()

        # Produce label list
        labels = []
        for label in self.shelf['classifiers']:
            # Use SVC model to classify mail for label l_id
            resp = self.shelf['classifiers'][label].predict(vecm)
            if resp > 0:
                labels.append(label)
        return labels

    def close(self):
        logging.info("Closing bank")
        self.shelf.close()
        self.tinydb.close()
