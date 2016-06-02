#!/usr/bin/env bash

wget https://www.cs.cornell.edu/people/pabo/movie-review-data/polarity_html.zip
rm -rf reviews_dataset movie
unzip polarity_html.zip
mv movie reviews_dataset
