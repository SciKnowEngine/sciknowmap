#
# Class to package up methods to query, load, and tag PubMed References
#
import os
from urllib.request import urlopen
from urllib.parse import quote_plus
from time import time,sleep
from sciknowmap.corpus import Corpus,Document
from sciknowmap.lx import SentTokenizer
from tqdm import tqdm
from bs4 import BeautifulSoup,Tag,Comment,NavigableString
from urllib.request import urlopen
from urllib.error import URLError
import re

class PubMedLoader:

    def __init__(self, path=None, corpus=None, num_topics=200, bigrams=False,
                 iters=1000, prefix=None):
        self.path = path
        self.tokenizer = SentTokenizer()

    #
    # loads all pubmed ids from a query.
    #
    def get_pmids_from_author_year_vol_page(self, author, year, vol, page):

        if( page != page or year != year or vol != vol):
            return None

        esearch_stem = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term='
        query = author + '[au] AND ' + year + '[dp] AND ' + vol + '[vi] AND ' + page + '[pg]'
        print(query)
        query_escaped = quote_plus(query)
        esearch_response = urlopen(esearch_stem + query_escaped)

        esearch_data = esearch_response.read().decode('utf-8')
        esearch_soup = BeautifulSoup(esearch_data, "lxml-xml")
        count_tag = esearch_soup.find('Count')
        if count_tag is None:
            raise Exception('Error returned from "' + query_escaped + '"')

        count = int(count_tag.string)

        if count > 1:
            print('Signature is ambiguous: "' + query + '"')
            return None

        if count == 0:
            return None

        start_time = time()
        latest_time = time()

        full_query = esearch_stem + query_escaped
        esearch_response = urlopen(full_query)
        esearch_data = esearch_response.read().decode('utf-8')
        esearch_soup = BeautifulSoup(esearch_data, "lxml-xml")
        pmid_tag = esearch_soup.find('Id')

        sleep(3)

        return pmid_tag.text

    #
    # loads all pubmed ids from a query.
    #
    def get_ids_from_pubmed_query(self, query, db="pubmed", oa=False, page_size=10000, time_threshold=0.3333334):

        if os.path.isfile(query):
            with open(query, 'r') as f:
                query = f.read()

        idPrefix = ''
        if oa:
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
            raise Exception('No Data returned from "' + query + '"')

        count = int(count_tag.string)
        start_time = time()
        latest_time = time()

        pmids = set()
        for i in tqdm(range(0,count,page_size)):
            full_query = esearch_stem + query + '&retstart=' + str(i)+ '&retmax=' + str(page_size)
            esearch_response = urlopen(full_query)
            esearch_data = esearch_response.read().decode('utf-8')
            esearch_soup = BeautifulSoup(esearch_data, "lxml-xml")
            for pmid_tag in esearch_soup.find_all('Id') :
                pmids.add(pmid_tag.text)
            delta_time = time() - latest_time
            if delta_time < time_threshold :
                sleep(time_threshold - delta_time)
        return pmids

    def get_pmids_of_papers_that_are_cited_by(self, pmid):
        url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi?' + \
              'dbfrom=pubmed&linkname=pubmed_pubmed_refs&id=' + pmid.strip()

        # ascii_url = iriToUri(url)
        pmids_of_papers_that_are_cited_by = set()
        pmid_response = urlopen(url)
        pmid_data = pmid_response.read().decode('utf-8')
        soup2 = BeautifulSoup(pmid_data, "lxml-xml")

        pmidtags = soup2.find_all('Id')
        for pmid_tag in pmidtags:
            if pmid_tag.text != pmid.strip():
                pmids_of_papers_that_are_cited_by.add(pmid_tag.text)
        return pmids_of_papers_that_are_cited_by

    def get_pmids_of_papers_that_cite(self, pmid):
        url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi?' + \
                'dbfrom=pubmed&linkname=pubmed_pubmed_citedin&id=' + pmid.strip()

        # ascii_url = iriToUri(url)
        pmids_of_papers_that_cite = set()
        pmid_response = urlopen(url)
        pmid_data = pmid_response.read().decode('utf-8')
        soup2 = BeautifulSoup(pmid_data, "lxml-xml")

        pmidtags = soup2.find_all('Id')
        for pmid_tag in pmidtags:
            if pmid_tag.text != pmid.strip():
                pmids_of_papers_that_cite.add(pmid_tag.text)
        return pmids_of_papers_that_cite

    def get_pmids_of_similar_papers_that_cite(self, pmid):
        url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi?' + \
                'dbfrom=pubmed&linkname=pubmed_pubmed&id=' + pmid.strip()

        # ascii_url = iriToUri(url)
        similar_papers = set()
        pmid_response = urlopen(url)
        pmid_data = pmid_response.read().decode('utf-8')
        soup2 = BeautifulSoup(pmid_data, "lxml-xml")

        pmidtags = soup2.find_all('Id')
        for pmid_tag in pmidtags:
            if pmid_tag.text != pmid.strip():
                similar_papers.add(pmid_tag.text)
        return similar_papers

    def convert_efetch_to_document(self, record):

        soup2 = BeautifulSoup(record, "lxml-xml")
        documents = {}

        for citation_tag in soup2.find_all('MedlineCitation'):

            pmid_tag = citation_tag.find('PMID')
            title_tag = citation_tag.find('ArticleTitle')
            abstract_tag = citation_tag.find('AbstractText')
            mesh_data = ",".join([x.text.replace('\n',' ') for x in citation_tag.findAll('MeshHeading')])
            authors_data = ",".join([x.text.replace('\n',' ') for x in citation_tag.findAll('LastName')])
            year = citation_tag.find('PubDate').find('Year')

            if pmid_tag is None or len(pmid_tag.text)==0 or title_tag is None or abstract_tag is None:
                continue

            is_review = "D"
            if True in [x.text == "Review" for x in citation_tag.findAll('PublicationType')]:
                is_review = "R"
            d = Document()
            d.id = pmid_tag.text
            d.title = title_tag.text
            d.sections = [{'text': self.tokenizer.tokenize(abstract_tag.text)}]
            d.mesh = mesh_data
            d.authors = authors_data
            d.book = ''
            d.references = []
            if year is not None:
                d.year = year.text
            else:
                d.year = "????"
            d.url = "https://www.ncbi.nlm.nih.gov/pubmed/" + d.id
            documents[d.id] = d

        return documents

    def get_docs_from_pmids(self, id_list):

        efetch_stem = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&retmode=xml&id='

        page_size = 100
        i = 0
        url = efetch_stem
        efetch_docs = {}
        for line in tqdm(id_list):
            try:
                l = re.split('\s+', line)
                pmid = l[0]
                i += 1
                if i >= page_size :
                    efetch_response = urlopen(url)
                    documents =self.convert_efetch_to_document(efetch_response.read().decode('utf-8'))
                    for d_id in documents:
                        d = documents[d_id]
                        efetch_docs[d.id] = d
                    url = efetch_stem
                    i = 0

                if re.search('\d$',url) :
                    url += ','
                url += pmid.strip()
            except URLError as e:
                sleep(10)
                print("URLError({0}): {1}".format(e.errno, e.strerror))
        sleep(3)
        efetch_response = urlopen(url)
        documents = self.convert_efetch_to_document(efetch_response.read().decode('utf-8'))
        for d_id in documents:
            d = documents[d_id]
            efetch_docs[d.id] = d

        return efetch_docs