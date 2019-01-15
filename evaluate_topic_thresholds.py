#!/usr/bin/env python3

# Topic Mapping Script
# Gully Burns

import math
import operator
import os
import pickle
import random
from datetime import datetime

import bokeh.plotting as bp
import click
import numpy as np
import pandas as pd
from bokeh.io import output_file, show
from bokeh.models import ColumnDataSource, HoverTool, TapTool, OpenURL
from bokeh.models import PanTool, BoxZoomTool, WheelZoomTool, ResetTool
from numpy.linalg import norm
from sklearn.manifold import TSNE
from tqdm import tqdm

from sciknowmap.mallet import Mallet
from sciknowmap.corpus import Corpus

#
# Provides HTML code for a single topic signature based on greyscale coding
# for each word
#
def topic_signature_html(m, t_tuple, n_words, colormap, global_min=None, global_max=None):
    t_id = t_tuple[0]
    t_percent = t_tuple[1]
    color = colormap[t_id]

    def invert_hex(hex_number):
        inverse = hex(abs(int(hex_number, 16) - 255))[2:]
        # If the number is a single digit add a preceding zero
        if len(inverse) == 1:
            inverse = '0' + inverse
        return inverse

    def float_to_greyscale(f):
        val = '%x' % int(f * 255)
        val = invert_hex(val)
        return '#%s%s%s' % (val, val, val)

    word_weights = sorted(
        m.topics[t_id].items(), key=operator.itemgetter(1), reverse=True
    )[:n_words]

    vals = [x[1] for x in word_weights]
    val_max = max(vals)
    val_min = math.sqrt(min(vals) / 2)
    val_diff = float(val_max - val_min)
    if global_min and global_max:
        global_diff = float(global_max - global_min)

    t_percent_2sf = '%s' % float('%.2g' % t_percent)

    ret = '<emph><font color="' + color + '">&#x25A0; </font>#' + str(t_id) + ' (' + t_percent_2sf + '): </emph>'

    for (y, z) in sorted(word_weights, key=lambda x: x[1],
                         reverse=True):

        p = float(z - val_min) / val_diff

        if global_min and global_max:
            q = float(z - global_min) / global_diff
        else:
            q = p

        ret += '<span style="color:%s" title="%s%% relevant">%s</span>\n' % (
            float_to_greyscale(p), int(q * 100), y.replace('_', '&nbsp;'))

    return ret

def document_signature_html(corpus, doc_id, DT, m, doc_list, n_topics, n_words, colormap):
    doc_count = DT.shape[0]
    top_topics = sorted(
        enumerate(DT[doc_id]), reverse=True, key=operator.itemgetter(1)
    )[:n_topics]

    doc = corpus[doc_list[doc_id]]
    html_signature = '<p><b>' + doc.title + '</b></br>'
    html_signature += '<i>' + ', '.join(doc.authors) + '</i>'
    # if(doc.url):
    #    html_signature += ' [<a href="'+doc.url+'">Link</a>]'
    html_signature += '</br>'
    html_signature += '</br>'.join([topic_signature_html(m, top_topics[i], n_words, colormap) for i in range(n_topics)])
    html_signature += '</p>'

    return html_signature
#
# SCRIPT TO RUN TOPIC MAPPING VISUALIZATION UNDER DIFFERENT METHODS
#
@click.command()
@click.argument('topicmodel_dir', type=click.STRING)
@click.argument('corpus_dir', type=click.Path(exists=True))
@click.argument('topic_names', type=click.Path(exists=True))
@click.argument('n_docs_per_topic', type=click.INT)
def main(topicmodel_dir, corpus_dir, topic_names, n_docs_per_topic):

    MALLET_PATH = '/usr/local/bin/mallet'

    corpus = Corpus(corpus_dir)
    m = Mallet(MALLET_PATH, topicmodel_dir, prefix=topicmodel_dir)

    td = []
    doc_list = [d_tuple[0] for d_tuple in m.topic_doc[0]]

    for (t, d_in_t_list) in enumerate(m.topic_doc):
        topic_counts = []
        topic_weights = []
        for (d, d_tuple) in enumerate(d_in_t_list):
            topic_counts.append(d_tuple[1])
        td.append(topic_counts)

    TD_raw = np.asarray(td)
    DT_raw = TD_raw.transpose()

    n_docs = DT_raw.shape[0]
    n_topics = DT_raw.shape[1]

    L1_norm = norm(DT_raw, axis=1, ord=1)
    DT = DT_raw / L1_norm.reshape(n_docs, 1)

    # Code to create the HTML display
    colors = []
    for i in range(200):
        r = lambda: random.randint(0,255)
        colors.append('#%02X%02X%02X' % (r(),r(),r()))

    colormap = np.array(colors)
    print(len(colormap))

    html_signatures = []
    for i in tqdm(range(n_docs)):
        html_signatures.append(document_signature_html(corpus, i, DT, m, doc_list, 5, 10, colormap))

    #display(HTML(html_signatures[0]))

    doc_count = DT.shape[0]
    doc_urls = [corpus[doc_list[i]].url for i in range(doc_count)]

    topic_keys = []
    for i in range(DT.shape[0]):
        topic_keys += DT[i].argmax(),

    # Load the names
    topic_names_tsv = pd.read_csv(topic_names, sep='\t')
    topic_names = {}
    for i, row in topic_names_tsv.iterrows():
        tid = row['topic']
        clarity = row['clarity']
        mixed= row['mixed']
        name = row['name']
        tag = row['tag']
        topic_names[tid] = (name, clarity, mixed, tag)

    html_string = """
            <html>
            <head>
            <title>Topic Evaluation</title>
            <style type="text/css">
            body {
                margin: 2em auto;
                font-family: 'Univers LT Std', 'Helvetica', sans-serif;
                max-width: 900px;
                width: 90%;
            }

            article {
                border-top: 4px solid #888;
                padding-top: 3em;
                margin-top: 3em;
            }

            section {
                padding-bottom: 3em;
                border-bottom: 4px solid #888;
                margin-bottom: 4em;
            }

            section section {
                border: 0px;
                padding: 0px;
                margin: 0em 0em 3em 0em;
            }

            h1 { font-size: 18pt; }

            h2 { font-size: 14pt; }

            label { margin-right: 6px; }

            input { margin-left: 6px; }

            div.topic {
                padding: 1em;
            }

            p.rate { font-weight: bold; margin-left: 2em; }

            blockquote { margin-left: 40px; }

            a { text-decoration: none; font-style: italic; border-bottom: 1px dotted grey; }

            a:hover { color: blue !important; }
            a:hover span { color: blue !important; }

            </style>
            </head>
            <body>
            <h1>Topic Evaluation</h1>
            <article>
    """

    for topic in range(0, DT.shape[1]):

        for document in range(0, n_docs_per_topic):
            html_string += '<section>\n'
            html_string += '    <section>\n'
            html_string += '<h2>Topic ' + str(topic) + '</h2>'
            html_string += '<div class="topic">'





if __name__ == '__main__':
    main()
