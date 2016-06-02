import shelve
import os


class DocumentBank:
    """
    DocumentBank manages the documents and operates the ML on it
    """
    def __init__(self, shelf_path, documents=None):
        """
        :param shelf_path: location of the db
        :type shelf_path: str
        :param documents: documents to add, None by default
        :type documents: list
        """
        self.path = shelf_path
        if documents is not None and len(documents) != 0:
            os.remove(shelf_path)
            self.shelf = shelve.open(self.path)
            self.shelf['documents'] = documents
        else:
            self.shelf = shelve.open(self.path)
            self.shelf['documents'] = []

    def add_documents(self, documents):
        """
        Add documents to the DocumentsBank
        :param documents: documents to add
        :type documents: list
        """
        self.shelf['documents'].extend(documents)
        self.shelf.sync()

    def close(self):
        """
        Closes the DocumentBank
        """
        self.shelf.close()
