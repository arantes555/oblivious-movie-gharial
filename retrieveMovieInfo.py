import requests
from urllib.parse import quote
import re


def retrieve_movie_info(title=None, movie_id=None, year=None):
    url_base = 'http://www.omdbapi.com/?'
    url = url_base
    if movie_id is not None:
        url += 'i=' + movie_id + '&'
    if title is not None:
        if year is None:
            if re.match(' \\([0-9]{4}\\)', title[-7:]):
                year = int(title[-5:-1])
                title = title[:-7]
        if year is not None:
            url += 'y=' + str(year) + '&'
        if title.endswith(', The'):
            title = 'The ' + title[:-5]
        title = quote(title)
        url += 't=' + title + '&'
    print(url)
    response = requests.get(url)
    return response.json()
