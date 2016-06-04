import config
from nltk import downloader

# Async, can't be run in main process :/
# for wordnet stemmer
downloader.download(info_or_id='wordnet', download_dir=config.NLTK_DATA_DIR)
# for snowball and porter stemmer
# downloader.download(info_or_id='punkt', download_dir=config.NLTK_DATA_DIR)
# stop words <- used in snowball
downloader.download(info_or_id='stopwords', download_dir=config.NLTK_DATA_DIR)
