
import codecs
import os.path
import argparse
from urllib.request import urlopen
from urllib.parse import quote_plus
from time import time,sleep
import sciknowmap.utils

from tqdm import tqdm
from bs4 import BeautifulSoup,Tag,Comment,NavigableString

def nmxl_files(members):
    try:
        for tarinfo in members:
            if os.path.splitext(tarinfo.name)[1] == ".nxml":
                yield tarinfo
    except IOError:
        doNothing = None

PAGE_SIZE = 10000
TIME_THRESHOLD = 0.3333334

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-q', '--query', help='PUBMED Query')
    parser.add_argument('-f', '--outFile', help='Output file')
    parser.add_argument('-d', '--db', help='Output file')
    sciknowmap.utils.add_boolean_argument(parser, 'oa')

    args = parser.parse_args()

    query = args.query

    if os.path.isfile(args.query):
        with open(args.query, 'r') as f:
            query = f.read()

    if os.path.isfile(args.outFile):
        os.remove(args.outFile)

    db = args.db
    if args.db is None:
        db = "pubmed"

    idPrefix = ''
    if args.oa:
        if db == 'PMC':
            query = '"open access"[filter] AND (' + query + ')'
            idPrefix = 'PMC'
        elif db == 'pubmed':
            query = '"loattrfree full text"[sb] AND (' + query + ')'

    esearch_stem = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db='+db+'&term='

    query = quote_plus(query)
    print(esearch_stem + query)
    esearch_response = urlopen(esearch_stem + query)
    esearch_data = esearch_response.read().decode('utf-8')
    esearch_soup = BeautifulSoup(esearch_data, "lxml-xml")
    count_tag = esearch_soup.find('Count')
    if count_tag is None:
        raise Exception('No Data returned from "' + args.query + '"')

    count = int(count_tag.string)

    f = codecs.open(args.outFile, 'w', 'utf-8')
    start_time = time()
    latest_time = time()

    print(count)

    for i in tqdm(range(0,count,PAGE_SIZE)):
        full_query = esearch_stem + query + '&retstart=' + str(i)+ '&retmax=' + str(PAGE_SIZE)
        esearch_response = urlopen(full_query)
        esearch_data = esearch_response.read().decode('utf-8')
        esearch_soup = BeautifulSoup(esearch_data, "lxml-xml")
        for pmid_tag in esearch_soup.find_all('Id') :
            f.write(idPrefix + pmid_tag.text + '\n')
        delta_time = time() - latest_time
        if delta_time < TIME_THRESHOLD :
            sleep(TIME_THRESHOLD - delta_time)
    f.close()
