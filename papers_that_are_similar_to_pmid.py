import re
import sys
import codecs
import os.path
import argparse
from urllib.request import urlopen
from tqdm import tqdm


from bs4 import BeautifulSoup,Tag,Comment,NavigableString
from elasticsearch import Elasticsearch, helpers

from robot_biocurator import pmc,utils,sd2
import re

def nmxl_files(members):
    try:
        for tarinfo in members:
            if os.path.splitext(tarinfo.name)[1] == ".nxml":
                yield tarinfo
    except IOError:
        doNothing = None

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inFile', help='Directory of robot-scraped files')
    parser.add_argument('-o', '--outFile', help='Output file')
    utils.add_boolean_argument(parser, 'reviews_only')

    args = parser.parse_args()

    with codecs.open(args.inFile, 'r', 'utf-8') as f:
        id_list = f.readlines()

    f = codecs.open(args.outFile, 'w', 'utf-8')

    for line in tqdm(id_list):

        l = re.split('\s+', line)
        pmid = l[0]
        if len(l)>1 and l[1] == 'False' and args.reviews_only:
            continue

        url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi?' + \
                'dbfrom=pubmed&linkname=pubmed_pubmed&id=' + pmid.strip()

        #ascii_url = iriToUri(url)
        pmid_response = urlopen(url)
        pmid_data = pmid_response.read().decode('utf-8')
        soup2 = BeautifulSoup(pmid_data, "lxml-xml")

        pmidtags = soup2.find_all('Id')
        for pmid_tag in pmidtags:
            if pmid_tag.text != pmid.strip():
                f.write(pmid_tag.text + '\t' + pmid.strip() + '\n')

    f.close()
