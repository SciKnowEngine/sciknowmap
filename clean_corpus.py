#!/usr/bin/env python3

import os
import re
import click
import html
from tqdm import tqdm

from utils.corpus import Corpus

@click.command()
@click.argument('corpusdir', type=click.Path(exists=True))
@click.argument('out-path', type=click.Path())
def main(corpusdir, out_path):

    if os.path.exists(out_path) is False:
        os.makedirs(out_path)

    corpus = Corpus(corpusdir)

    for dkey in tqdm(corpus.docs):
        doc = corpus.docs[dkey]
        for sec in doc.sections:
            text = []
            for sent in sec['text']:
                tmp = re.sub("[\\\\]+n", " ", sent)
                tmp = html.unescape(tmp)
                text.append(tmp)
            sec['text'] = text

    corpus.export(out_path)

if __name__ == '__main__':
    main()
