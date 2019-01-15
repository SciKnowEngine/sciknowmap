from elasticsearch import Elasticsearch, helpers
import re
import sys
import codecs
import os.path
import argparse
from urllib.request import urlopen
from tqdm import tqdm
from bs4 import BeautifulSoup,Tag,Comment,NavigableString
from sciknowmap.corpus import Corpus, Document
from pathlib import Path


def nmxl_files(members):
    try:
        for tarinfo in members:
            if os.path.splitext(tarinfo.name)[1] == ".nxml":
                yield tarinfo
    except IOError:
        doNothing = None


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inDir', help='Corpus Directory')

    args = parser.parse_args()

    list_of_files = {}
    for (dirpath, dirnames, filenames) in os.walk(args.inDir):
        for filename in filenames:
            if filename.endswith('.json'):
                list_of_files[filename] = os.sep.join([dirpath, filename])

    corpus = Corpus(args.inDir)
    corpus.remove_duplicates()

    for filename in tqdm(list_of_files):
        dkey = filename.replace(".json","")
        if corpus.docs.get(dkey, None) is None:
            os.remove(list_of_files.get(filename))
