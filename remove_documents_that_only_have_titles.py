from elasticsearch import Elasticsearch, helpers
import re
import sys
import codecs
import os.path
import argparse
from urllib.request import urlopen
from tqdm import tqdm
from bs4 import BeautifulSoup,Tag,Comment,NavigableString
from utils.corpus import Corpus, Document
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
    parser.add_argument('-i', '--inDir', help='Directory of robot-scraped files')

    args = parser.parse_args()

    list_of_files = {}
    for (dirpath, dirnames, filenames) in os.walk(args.inDir):
        for filename in filenames:
            if filename.endswith('.json'):
                list_of_files[filename] = os.sep.join([dirpath, filename])

    corpus = Corpus(args.inDir)
    for dkey in tqdm(corpus.docs):
        doc = corpus.docs[dkey]
        doc_lang = doc.detect_lang()

        if len(doc.sections) == 0 :
            file_path = list_of_files[doc.id + '.json']
            print('\n---- Removing %s' %(file_path))
            os.remove(file_path)