import shelve
import os
import logging
from sklearn.feature_extraction.text import CountVectorizer


class DocumentBank:
    """
    DocumentBank manages the documents and operates the ML on it
    """

    def __init__(self, shelf_path, documents=None, stop_words=None, max_words=2000):
        """
        :param shelf_path: location of the db
        :type shelf_path: str
        :param documents: documents to add, None by default
        :type documents: list
        """
        self.path = shelf_path
        self.max_words = max_words
        if documents is not None:
            logging.info('documents were provided, removing database if any')
            os.remove(shelf_path)
            self.shelf = shelve.open(self.path, writeback=True)
            self.shelf['documents'] = documents
        else:
            logging.info("no documents were provided, loading database")
            self.shelf = shelve.open(self.path, writeback=True)
            self.shelf['documents'] = []

        if stop_words is not None:
            self.shelf['stop_words'] = stop_words
        else:
            self.shelf['stop_words'] = []

        self.shelf.sync()

    def add_documents(self, documents):
        """
        Add documents to the DocumentsBank
        :param documents: documents to add
        :type documents: list
        """
        logging.debug("adding specified documents to database")
        self.shelf['documents'].extend(documents)
        self.shelf.sync()

    def vectorize(self):
        self.shelf['vectorized_documents'] = CountVectorizer(decode_error='ignore',
                                                             strip_accents='unicode',
                                                             min_df=4,
                                                             max_df=0.98,
                                                             stop_words=self.shelf['stop_words'],
                                                             max_features=self.max_words)

        def corpus():
            for document in list(self.shelf['documents']):
                yield document['content']

        features_matrix = self.shelf['vectorized_documents'].fit_transform(corpus())

        self.shelf['features_matrix'] = features_matrix

        # Inverse the vectorized vocabulary
        self.shelf.data['dictionnary'] = self.shelf.data['vectorized_documents'].get_feature_names()
        self.shelf.sync()

    def close(self):
        """
        Closes the DocumentBank
        """
        logging.info("Closing bank")
        self.shelf.close()
