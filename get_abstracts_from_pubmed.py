from elasticsearch import Elasticsearch, helpers
from robot_biocurator import pmc, utils, sd2
import re
import sys
import codecs
import os.path
import argparse
from urllib.request import urlopen
from tqdm import tqdm
from bs4 import BeautifulSoup,Tag,Comment,NavigableString

def nmxl_files(members):
    try:
        for tarinfo in members:
            if os.path.splitext(tarinfo.name)[1] == ".nxml":
                yield tarinfo
    except IOError:
        doNothing = None


if __name__ == '__main__':

    efetch_stem = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&retmode=xml&id='

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inFile', help='Directory of robot-scraped files')
    parser.add_argument('-o', '--outFile', help='Output file')

    args = parser.parse_args()
    if os.path.isfile(args.outFile):
        os.remove(args.outFile)

    with codecs.open(args.inFile, 'r', 'utf-8') as f:
        id_list = f.readlines()

    f = codecs.open(args.outFile, 'w', 'utf-8')

    page_size = 100
    i = 0
    url = efetch_stem
    efetch_data = []
    for line in tqdm(id_list):
        l = re.split('\s+', line)
        pmid = l[0]
        i += 1
        if i == page_size :
            efetch_response = urlopen(url)
            efetch_data.append( efetch_response.read().decode('utf-8'))
            url = efetch_stem
            i = 0

        if re.search('\d$',url) :
            url += ','
        url += pmid.strip()

    efetch_response = urlopen(url)
    efetch_data.append(efetch_response.read().decode('utf-8'))
    url = efetch_stem

    print("\n\nSaving records to output: " + args.outFile)

    for record in tqdm(efetch_data):
        soup2 = BeautifulSoup(record, "lxml-xml")

        for citation_tag in soup2.find_all('MedlineCitation') :

            pmid_tag = citation_tag.find('PMID')
            title_tag = citation_tag.find('ArticleTitle')
            abstract_tag = citation_tag.find('AbstractText')
            review_tags = citation_tag.findAll('PublicationType')

            if pmid_tag is None or title_tag is None or abstract_tag is None:
                continue

            is_review = False
            if True in [x.text == "Review" for x in citation_tag.findAll('PublicationType')]:
                is_review = True

            f.write(pmid_tag.text + '\t' + str(is_review) + '\t' + title_tag.text +'\t' + abstract_tag.text + '\n')

    f.close()
