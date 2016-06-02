#!/usr/bin/env bash

rm -rf tmp
mkdir tmp
rm -rf resources/reviews_dataset
cd tmp
wget https://www.cs.cornell.edu/people/pabo/movie-review-data/polarity_html.zip
unzip polarity_html.zip
cd ..
mkdir resources
mv tmp/movie resources/reviews_dataset
rm -f polarity_html.zip
rm -rf tmp
