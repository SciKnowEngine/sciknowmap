
import argparse

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utils.google_books_corpus import read_resources_from_dir

# Following https://developers.google.com/books/docs/v1/getting_started?csw=1
# We run queries over all terms in the basic taxonomy against the Google Books Volume collection
# and convert them to the Erudite Learning Resource schema


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inDir', help='Input dir of json files')
    parser.add_argument('-o', '--outDir', help='Output directory')

    args = parser.parse_args()

    resources = read_resources_from_dir(args.inDir)

    # BASIC STATS
    print('\n\tNumber of Unique Books: {0:d}.'.format(len(resources)))

    free_book_count = 0
    for b in resources.values():
        if(b.get('price', None) is not None):
            if(b['price']['amount'] == 0.0):
                free_book_count += 1
    print('\tNumber of Free Books: {0:d}.'.format(free_book_count))

    #
    # Write histograms for text length in documents
    #
    df1 = pd.DataFrame(resources.values())
    text_df = pd.DataFrame.from_records(df1, columns=['description', 'title'])
    lengths = df1.apply(lambda row: len(row['description']) + len(row['title']), axis=1)
    text_df['lengths'] = lengths
    sns.set(color_codes=True)
    sns.distplot(lengths)

    plt.show()
