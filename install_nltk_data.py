import config
from nltk import downloader

# Async, can't be run in config :/
downloader.download(info_or_id='plunkt', download_dir=config.NLTK_DATA_DIR)
downloader.download(info_or_id='stopwords', download_dir=config.NLTK_DATA_DIR)
