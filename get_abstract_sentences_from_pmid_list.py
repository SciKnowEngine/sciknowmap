import re
import sys
import codecs
import os.path
import argparse
import time
from urllib.request import urlopen
from urllib.error import URLError
from tqdm import tqdm
from bs4 import BeautifulSoup,Tag,Comment,NavigableString
import nltk
nltk.download('punkt')
from nltk import word_tokenize, sent_tokenize

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def nmxl_files(members):
    try:
        for tarinfo in members:
            if os.path.splitext(tarinfo.name)[1] == ".nxml":
                yield tarinfo
    except IOError:
        doNothing = None

def write_text_from_medline_record_to_disk(record, f):

    soup2 = BeautifulSoup(record, "lxml-xml")

    for citation_tag in soup2.find_all('MedlineCitation'):

        pmid_tag = citation_tag.find('PMID')
        title_tag = citation_tag.find('ArticleTitle')
        abstract_tag = citation_tag.find('AbstractText')
        review_tags = citation_tag.findAll('PublicationType')

        if pmid_tag is None or title_tag is None or abstract_tag is None:
            continue

        is_review = False
        if True in [x.text == "Review" for x in citation_tag.findAll('PublicationType')]:
            is_review = True

        sentences = [title_tag.text]
        for sent in sent_tokenize(abstract_tag.text):
            sentences.append(sent)

        for i, sent in enumerate(sentences):
            f.write(pmid_tag.text + '\t' +
                    's'+str(i) + '\t' +
                    str(is_review) + '\t' +
                    " ".join(word_tokenize(sent)) + '\n')

if __name__ == '__main__':

    efetch_stem = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&retmode=xml&id='

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inFile', help='Directory of robot-scraped files')
    parser.add_argument('-o', '--outFile', help='Output file')

    args = parser.parse_args()

    eprint(args.inFile)
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
        try:
            l = re.split('\s+', line)
            pmid = l[0]
            i += 1
            if i >= page_size :
                efetch_response = urlopen(url)
                write_text_from_medline_record_to_disk(efetch_response.read().decode('utf-8'), f)
                url = efetch_stem
                i = 0

            if re.search('\d$',url) :
                url += ','
            url += pmid.strip()
        except URLError as e:
            time.sleep(10)
            print("URLError({0}): {1}".format(e.errno, e.strerror))


    efetch_response = urlopen(url)
    write_text_from_medline_record_to_disk(efetch_response.read().decode('utf-8'), f)
