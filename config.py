import os

# Minimum relevance (in per cent of the total amount of documents) to accept a classifier
MIN_RELEVANCE = 0.001

# Amount of reviews to analyse to train the classifiers
MAX_REVIEWS = 20000

# Movies to classify after the model is trained
MOVIES_TO_CLASSIFY = 20

# Maximum amount of words in the dictionnary
MAX_FEATURES = 1000

# Amount of topics to extract, keep it relatively low
N_TOPICS = 20

# Amount of words to display for each topic, doesn't affect anything except printing
N_TOP_WORDS = 15

# Parameter that controls spareness
BETA = 1e-4

AMAZON_REVIEWS_FILE = './resources/Movies_and_TV_5.json'

METADATA_FILE = './resources/meta_Movies_and_TV.json'

LANGUAGE_STOP_WORDS_PATH = './resources/stopwords/english'

PROJECT_STOP_WORDS_PATH = './resources/movies_stopwords'

NLTK_DATA_DIR = os.path.abspath('./resources/nltk_data/')

os.environ['NLTK_DATA'] = NLTK_DATA_DIR

# Amount of parallel jobs the computer can take (core amount x2 with hyper-threading)
N_JOBS = 8
