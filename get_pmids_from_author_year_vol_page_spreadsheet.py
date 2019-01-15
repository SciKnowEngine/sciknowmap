
import codecs
import os.path
import argparse
from urllib.request import urlopen
from urllib.parse import quote_plus
from time import time,sleep
import sciknowmap.utils
import pandas as pd

from sciknowmap.loader import PubMedLoader

from tqdm import tqdm
from bs4 import BeautifulSoup,Tag,Comment,NavigableString

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inFile', help='Input file')
    parser.add_argument('-o', '--outFile', help='Output file')
    args = parser.parse_args()

    if os.path.isfile(args.inFile) is False:
        raise Exception('No input file:' + args.inFile)

    df_au_dp_vi_pg = pd.read_csv(args.inFile, sep='\t', dtype={'author':str,'year':str,'vol':str,'page':str,'pmid':str,'title':str})
    l = PubMedLoader()

    pmids = []
    for i, row in df_au_dp_vi_pg.iterrows():
        au = row.get('author')
        dp = row.get('year')
        vi = row.get('vol')
        pg = row.get('page')
        ti = row.get('title')
        pmid = row.get('pmid')
        if pmid != pmid:
            pmid = l.get_pmids_from_author_year_vol_page(au,dp,vi,pg)
            if pmid is not None:
                pmids.append(pmid)
        else:
            pmids.append(pmid)

    df = pd.DataFrame({'pmids': pmids})
    df.to_csv(args.outFile, sep='\t')
