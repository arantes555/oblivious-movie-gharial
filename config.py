import os

# Minimum relevance (in per cent of the total amount of documents) to accept a classifier
MIN_RELEVANCE = 0.001

# Amount of reviews to analyse to train the classifiers
MAX_REVIEWS = 2000

# Movies to classify after the model is trained
MOVIES_TO_CLASSIFY = 10

# Maximum amount of words in the dictionnary
MAX_FEATURES = 2000

# Amount of topics to extract, keep it relatively low
N_TOPICS = 10

# Amount of words to display for each topic, doesn't affect anything except printing
N_TOP_WORDS = 12

HTML_REVIEWS_DIR = './resources/reviews_dataset/'

AMAZON_REVIEWS_FILE = './resources/movies.txt'

LANGUAGE_STOP_WORDS_PATH = './resources/stopwords/english'

PROJECT_STOP_WORDS_PATH = './resources/movies_stopwords'

NLTK_DATA_DIR = os.path.abspath('./resources/nltk_data/')

os.environ['NLTK_DATA'] = NLTK_DATA_DIR

# Amount of parallel jobs the computer can take (core amount x2 with hyper-threading)
N_JOBS = 8
