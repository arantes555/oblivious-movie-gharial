import os

# Amount of reviews to analyse to train the classifiers
MAX_REVIEWS = 100000

# Movies to classify after the model is trained
MOVIES_TO_CLASSIFY = 100

# Maximum amount of words in the dictionnary
MAX_FEATURES = 15000

N_TOP_WORDS = 12

HTML_REVIEWS_DIR = './resources/reviews_dataset/'

AMAZON_REVIEWS_FILE = './resources/movies.txt'

LANGUAGE_STOP_WORDS_PATH = './resources/stopwords/english'

PROJECT_STOP_WORDS_PATH = './resources/movies_stopwords'

NLTK_DATA_DIR = os.path.abspath('./resources/nltk_data/')

os.environ['NLTK_DATA'] = NLTK_DATA_DIR


