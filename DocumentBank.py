import shelve
import logging
from sklearn.feature_extraction.text import CountVectorizer
from tinydb import TinyDB
from tinydb.storages import JSONStorage
from tinydb.middlewares import CachingMiddleware
import utils
import warnings
from numpy import array, argsort, zeros
import sklearn.feature_extraction.text as sktext
from time import time
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

    def add_document(self, document_content, document_metadata):
        self.tinydb.insert({'content': document_content, 'metadata': document_metadata})

    def vectorize(self, stop_words=None, max_features=2000):
        logging.info('Starting vectorizing...')
        self.shelf['vectorized_documents'] = CountVectorizer(decode_error='ignore',
                                                             strip_accents='unicode',
                                                             min_df=10,
                                                             max_df=0.80,
                                                             stop_words=stop_words,
                                                             max_features=max_features
        )

        def corpus():
            for document in self.tinydb.all():
                yield document['content']

        features_matrix = self.shelf['vectorized_documents'].fit_transform(corpus())

        self.shelf['features_matrix'] = features_matrix

        # Inverse the vectorized vocabulary
        self.shelf['dictionnary'] = self.shelf['vectorized_documents'].get_feature_names()
        logging.info('Vectorizing done')
        self.shelf.sync()

    def topic_extraction(self, options=None):
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
        return bas, fctr_res.basis().toarray()  # Return (H,W) matrix factors

    def close(self):
        logging.info("Closing bank")
        self.shelf.close()
        self.tinydb.close()
