from html.parser import HTMLParser
import re
import sys
from time import time
import logging


class ParsingError(Exception):
    pass


# create a subclass and override the handler methods
class HtmlReviewParser(HTMLParser):
    def error(self, message):
        pass

    def __init__(self):
        self.title = ''
        self.reviewer = ''
        self.capsule_review = ''
        self.review = ''
        self.review_paragraphs = []
        self.rating = 0

        self._inBody = False
        self._inH1 = False
        self._inH3 = False
        self._inReview = False

        self._gotHeader = False
        self._gotFooter = False

        super().__init__()

    def handle_starttag(self, tag, attrs):
        if tag == 'body':
            self._inBody = True
        elif self._inBody:
            if tag == 'h1':
                self._inH1 = True
            elif tag == 'h3':
                self._inH3 = True
            elif self._gotHeader and not self._gotFooter:
                self._inReview = True

    def handle_endtag(self, tag):
        if tag == 'body':
            self._inBody = False
        elif self._inBody:
            if tag == 'h1':
                self._inH1 = False
            if tag == 'h3':
                self._inH3 = False
            elif tag == 'pre' and not self._gotHeader:
                self._gotHeader = True
            elif tag == 'pre' and self._gotHeader:
                self._gotFooter = True
            elif tag == 'p' and self._gotHeader and not self._gotFooter:
                self._inReview = False

    def handle_data(self, data):
        if self._inBody:
            if self._inH1:
                self.title = data
            elif self._inH3:
                if not data == 'reviewed by':
                    self.reviewer += data
            elif self._inReview:
                if not self.capsule_review:
                    self.capsule_review = data.replace('          Capsule review:  ', '', 1).replace('\n     ', ' ')
                else:
                    self.review += data + '\n'

    def parse_review(self):
        split_rating = re.split('[\\w\\s]+([\\-\\+]?[0-4])[\\w\\s]+\\-4 to \\+4', self.review)
        self.review = split_rating[0]
        try:
            self.rating = split_rating[1]
        except IndexError:
            self.rating = '?'
        return {
            'movie': self.title,
            'reviewer': self.reviewer,
            'review': self.review,
            'capsule_review': self.capsule_review,
            'rating': self.rating
        }

    @staticmethod
    def parse(review_html):
        parser = HtmlReviewParser()
        parser.feed(review_html)
        return parser.parse_review()


class AmazonReviewsParser:
    @staticmethod
    def parse_review(string):
        string = string.replace('<br />', ' ') \
            .replace('<p>', '').replace('<p/>', '') \
            .replace('<p >', '').replace('<p /', '')
        parts = [part.split(': ')[1] for part in string.split('\n')[0:8]]
        review = {
            'movie_id': parts[0],
            'reviewer_id': parts[1],
            'reviewer': parts[2],
            'helpfulness': parts[3],
            'score': parts[4],
            'time': parts[5],
            'summary': parts[6],
            'review': parts[7]
        }
        return review

    @staticmethod
    def from_file(file, max_reviews=sys.maxsize):
        max_reviews = max_reviews or sys.maxsize
        t0 = time()
        last_t = t0
        fail = 0
        movies = {}
        n_reviews = 0
        temp = ''
        with open(file, encoding='latin-1') as f:
            while n_reviews < max_reviews:
                if time() - last_t > 10:
                    last_t = time()
                    logging.info('%i reviews read for %i movies, %i failed, in %is.'
                                 % (n_reviews, len(movies), fail, time() - t0))
                temp2 = f.read(200)
                if temp2 == '':
                    break
                temp += temp2
                while '\n\n' in temp:
                    review, temp = temp.split('\n\n', maxsplit=1)
                    try:
                        review = AmazonReviewsParser.parse_review(review)
                        movie_id = review.pop('movie_id')
                        if movie_id not in movies:
                            movies[movie_id] = []
                        movies[movie_id].append(review)
                        n_reviews += 1
                    except IndexError:
                        fail += 1
        logging.info('Done : %i reviews read for %i movies, %i failed, in %is.'
                     % (n_reviews, len(movies), fail, time() - t0))
        return n_reviews, movies

    @staticmethod
    def parse_metadata(file):
        logging.info('Reading metadata file ...')
        t0 = time()
        products = {}
        with open(file) as f:
            for l in f:
                metadata = eval(l)
                product_id = metadata.pop('asin')
                products[product_id] = metadata
        logging.info('%i products read in %is.' % (len(products), time() - t0))
        return products

    @staticmethod
    def from_json(file, meta=None, max_reviews=sys.maxsize, max_movies=sys.maxsize):
        meta = AmazonReviewsParser.parse_metadata(meta) if meta is not None else None
        max_reviews = max_reviews or sys.maxsize
        max_movies = max_movies or sys.maxsize
        t0 = time()
        last_t = t0
        fail = 0
        movies = {}
        n_reviews = 0
        n_movies_with_title = 0
        not_movies = {}

        def log(done=False):
            n_movies = len(movies)
            n_not_movies = len(not_movies)
            n_reviews_not_movies = sum(not_movies.values())
            logging.info(('Done: ' if done else '') + '%i reviews read for %i movies (%i%% with titles,'
                         ' %i%% of products for %i%% of reviews were not movies), %i failed, in %is.'
                         % (n_reviews,
                            n_movies,
                            n_movies_with_title * 100 / n_movies,
                            n_not_movies * 100 / (n_movies + n_not_movies),
                            n_reviews_not_movies * 100 / (n_reviews_not_movies + n_reviews),
                            fail,
                            time() - t0))

        logging.info('Reading reviews ...')
        with open(file) as f:
            for l in f:
                if n_reviews >= max_reviews \
                        or len(movies) >= max_movies:
                    break
                if time() - last_t > 10:
                    last_t = time()
                    log()
                try:
                    review = eval(l)
                    movie_id = review.pop('asin')
                    movie_meta = None
                    if movie_id not in movies:
                        if meta is not None:
                            movie_meta = meta[movie_id]
                            if 'Movies' not in movie_meta['categories'][0]:
                                not_movies[movie_id] = not_movies[movie_id] + 1 if movie_id in not_movies else 1
                                raise ParsingError('Not a movie')
                        movies[movie_id] = {
                            'title': movie_meta['title'] if movie_meta and 'title' in movie_meta else '',
                            'reviews': []
                        }
                        if movie_meta and 'title' in movie_meta:
                            n_movies_with_title += 1
                    movies[movie_id]['reviews'].append({
                        'reviewer_id': review['reviewerID'],
                        'reviewer': review['reviewerName'] if 'reviewerName'
                                                              in review else 'Unnamed' + review['reviewerID'],
                        'helpfulness': review['helpful'],
                        'score': review['overall'],
                        'time': review['unixReviewTime'],
                        'summary': review['summary'],
                        'review': review['reviewText']
                    })
                    n_reviews += 1
                except KeyError as e:
                    fail += 1
                    logging.info('Fail !\nData:' + l + '\nError:\n' + str(e))
                except ParsingError:
                    pass
        log(done=True)
        return n_reviews, movies
