
import codecs
import os.path
import argparse
from urllib.request import urlopen
from urllib.parse import quote_plus
from time import time,sleep
import sciknowmap.utils
from sciknowmap.loader import PubMedLoader
from sciknowmap.corpus import Corpus,Document
from time import gmtime, strftime

from tqdm import tqdm
import time
from bs4 import BeautifulSoup,Tag,Comment,NavigableString

def nmxl_files(members):
    try:
        for tarinfo in members:
            if os.path.splitext(tarinfo.name)[1] == ".nxml":
                yield tarinfo
    except IOError:
        doNothing = None

def add_tagged_doc_to_corpus(c, doc_lookup, d_id, tag):
    d = c.get_document(d_id)
    if d is None:
        d = doc_lookup[d_id]
        d.tags.append(tag)
        c.add(d)
    else:
        d.tags.append(tag)

PAGE_SIZE = 10000
TIME_THRESHOLD = 0.3333334

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-q', '--query', help='PUBMED Query')
    parser.add_argument('-f', '--outDir', help='Output Directory')
    parser.add_argument('-l', '--logDir', help='Output File')
    sciknowmap.utils.add_boolean_argument(parser, 'pubmed')
    sciknowmap.utils.add_boolean_argument(parser, 'oa')
    sciknowmap.utils.add_boolean_argument(parser, 'g__citing_papers')
    sciknowmap.utils.add_boolean_argument(parser, 'd__cited_papers')

    args = parser.parse_args()

    queries = args.query.split('\n')

    if os.path.isfile(args.query):
        with open(args.query, 'r') as f:
            queries= f.read().split('\n')

    if os.path.isfile(args.outDir):
        os.remove(args.outDir)

    timesig = strftime("%Y-%m-%d_%H:%M:%S", gmtime())
    log = open(args.logDir + '/log_' + timesig + '.txt', 'w')
    log.write("Query\tAuthored Paper Count\n")

    db = "PMC"
    if args.pubmed:
        db = "pubmed"

    loader = PubMedLoader()

    c = Corpus()
    for query in queries:

        time.sleep(3)

        # if the format of the query line is '#label <pmids>
        # just read the pmids in the query form.
        if query[:1] == '#':
            split_line = query.split(sep='\t')
            query = split_line[0][1:]
            base_pmids = split_line[1:]
        else:
            base_pmids = loader.get_ids_from_pubmed_query(query, db, args.oa)
        log.write("%s\t%d\n" % (query, len(base_pmids)))
        base_documents = loader.get_docs_from_pmids(base_pmids)
        for d_id in base_documents:
            add_tagged_doc_to_corpus(c, base_documents, d_id, 'BASE: ' + query)

        citing_pmids = set()
        if args.g__citing_papers:
            for pmid in tqdm(base_pmids):
                time.sleep(3)
                temp = loader.get_pmids_of_papers_that_cite(pmid)
                for temp_pmid in temp:
                    citing_pmids.add(temp_pmid)
        citing_documents = loader.get_docs_from_pmids(citing_pmids)
        for d_id in citing_documents:
            add_tagged_doc_to_corpus(c, citing_documents, d_id, 'CITING: ' + query)

        cited_pmids = set()
        if args.d__cited_papers:
            for pmid in tqdm(cited_pmids):
                time.sleep(3)
                temp = loader.get_pmids_of_papers_that_are_cited_by(pmid)
                for temp2_pmid in temp:
                    cited_pmids.add(temp2_pmid)
        cited_documents = loader.get_docs_from_pmids(cited_pmids)
        for d_id in cited_documents:
            add_tagged_doc_to_corpus(c, cited_documents, d_id, 'CITED: ' + query)

    c.export(args.outDir,abstract=True)
    log.close()
