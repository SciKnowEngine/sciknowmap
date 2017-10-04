#!/usr/bin/env python3

# TechKnAcq: Concept Graph
# Jonathan Gordon

#
# This script runs MALLET on a document corpus to generate the needed output files for constructing a
# Concept Graph.
#

import os
import random

import click

from utils import Mallet
from utils.corpus import Corpus

# Parameters

MALLET_PATH = '/usr/local/bin/mallet'

LDA_TOPICS = 300
LDA_ITERATIONS = 200

@click.command()
@click.argument('corpusdir', type=click.Path(exists=True))
@click.argument('out-path', type=click.Path(exists=True))
@click.argument('num-topics', default=LDA_TOPICS)
@click.argument('num-iterations', default=LDA_ITERATIONS)
def main(corpusdir, out_path, num_topics, num_iterations):

    rand_prefix = hex(random.randint(0, 0xffffff))[2:]
    prefix = os.path.join(out_path, rand_prefix)

    corpus = Corpus(corpusdir)
    # corpus.fix_text()

    print('Generating topic model.')
    mallet_corpus = prefix + '/corpus'
    os.makedirs(mallet_corpus)
    corpus.export(mallet_corpus, abstract=False, form='text')
    model = Mallet(MALLET_PATH, mallet_corpus, prefix=prefix, num_topics=num_topics,
                   iters=num_iterations, bigrams=False)

if __name__ == '__main__':
    main()
