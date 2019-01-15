import re
import sys
import codecs
import os.path
import argparse
from urllib.request import urlopen
from tqdm import tqdm

from bs4 import BeautifulSoup,Tag,Comment,NavigableString

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
    parser.add_argument('-i', '--inFile', help='File of input pmids')
    parser.add_argument('-o', '--outFile', help='Output file')

    args = parser.parse_args()

    with codecs.open(args.inFile, 'r', 'utf-8') as f:
        id_list = f.readlines()

    f = codecs.open(args.outFile, 'w', 'utf-8')

    for line in tqdm(id_list):

        l = re.split('\s+', line)
        pmid = l[0]

        url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi?' + \
                'dbfrom=pubmed&linkname=pubmed_pubmed_citedin&id=' + pmid.strip()

        pmid_response = urlopen(url)
        pmid_data = pmid_response.read().decode('utf-8')
        soup2 = BeautifulSoup(pmid_data, "lxml-xml")

        pmidtags = soup2.find_all('Id')
        for pmid_tag in pmidtags:
            f.write(pmid_tag.text + '\t' + pmid.strip() + '\n')

    f.close()




