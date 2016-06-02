import shelve
import logging
from sklearn.feature_extraction.text import CountVectorizer
from tinydb import TinyDB
from tinydb.storages import JSONStorage
from tinydb.middlewares import CachingMiddleware
import utils


class DocumentBank:
    """
    DocumentBank manages the documents and operates the ML on it
    """

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

    def vectorize(self):
        logging.info('Starting vectorizing...')
        self.shelf['vectorized_documents'] = CountVectorizer(decode_error='ignore',
                                                             strip_accents='unicode',
                                                             min_df=0.02,
                                                             max_df=0.98)

        def corpus():
            for document in self.tinydb.all():
                yield document['content']

        features_matrix = self.shelf['vectorized_documents'].fit_transform(corpus())

        self.shelf['features_matrix'] = features_matrix

        # Inverse the vectorized vocabulary
        self.shelf['dictionnary'] = self.shelf['vectorized_documents'].get_feature_names()
        logging.info('Vectorizing done')
        self.shelf.sync()

    def close(self):
        """
        Closes the DocumentBank
        """
        logging.info("Closing bank")
        self.shelf.close()
        self.tinydb.close()
