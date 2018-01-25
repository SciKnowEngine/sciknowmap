from bottle import Bottle
from bottle import route, run
from bottle import get, post, request  # or route

import math
from utils.mallet import Mallet
from utils.corpus import Corpus
import os
import warnings
import sys
import codecs
import numpy
import argparse
import json
import pickle
import click
from numpy.linalg import norm
import numpy as np
import operator
from tqdm import tqdm
import random
import pandas as pd

from sciknowmap_server_plugin import SKMPlugin

#
# Provides HTML code for a single topic signature based on greyscale coding
# for each word
#
def topic_signature_html(m, t_id, n_words, colormap, global_min=None, global_max=None, t_percent=None):
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

    ret = '<emph><font color="' + color + '">&#x25A0; </font>#' + str(t_id) + ': </emph>'
    if( t_percent is not None ) :
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

def document_topic_signature_html(corpus, doc_id, DT, m, doc_list, n_words, n_topics, colormap):
    doc_count = DT.shape[0]

    # (t_id, t_percent) tuples
    top_topics = sorted(
        enumerate(DT[doc_id]), reverse=True, key=operator.itemgetter(1)
    )[:n_topics]

    doc = corpus[doc_list[doc_id]]
    html_signature = '<p><b>' + doc.title + '</b></br>'
    html_signature += '<i>' + ', '.join(doc.authors) + '</i>'
    # if(doc.url):
    #    html_signature += ' [<a href="'+doc.url+'">Link</a>]'
    html_signature += '</br>'
    html_signature += '</br>'.join(
        [topic_signature_html(m, top_topics[i][0], n_words, colormap, t_percent=top_topics[i][1]) for i in range(n_topics)])
    html_signature += '</p>'

    return html_signature


def topic_document_signature_html(corpus, t_id, TD, m, doc_list, n_docs, colormap):

    # (t_id, t_percent) tuples
    top_documents = sorted(
        enumerate(TD[t_id]), reverse=True, key=operator.itemgetter(1)
    )[:n_docs]

    html_signature = "<ol>"
    for td in top_documents:
        doc_id = td[0]
        doc = corpus[doc_list[doc_id]]
        html_signature += '<li>'
        html_signature += '<b>(' + "{0:.4f}".format(td[1]) + ')</b> '
        html_signature += '<i>' + ', '.join(doc.authors) + '</i> '
        if (doc.url):
            html_signature += '<a href="' + str(doc.url) + '">'
        html_signature +=  doc.title
        if(doc.url):
            html_signature += '</a>'
        html_signature += '</li>'
    html_signature += '</ol>'

    return html_signature

@click.command()
@click.argument('topicmodel_dir', type=click.STRING)
@click.argument('corpus_dir', type=click.Path(exists=True))
@click.argument('port', type=click.INT)
@click.argument('n_docs_per_topic', type=click.INT)
def main(topicmodel_dir, corpus_dir, port, n_docs_per_topic):

    skm = SKMPlugin()
    app = Bottle()
    app.install(skm)

    #
    # Loads the topic model
    #
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
    TD = DT.transpose()

    # Code to create the HTML display
    colors = []
    for i in range(200):
        r = lambda: random.randint(0,255)
        colors.append('#%02X%02X%02X' % (r(),r(),r()))

    colormap = np.array(colors)
    print(len(colormap))

    # Load the names
    eval_file_path = topicmodel_dir + "/topic_names.tsv"
    topic_names = []
    if os.path.exists(eval_file_path):
        topic_names_tsv = pd.read_csv(eval_file_path, sep='\t')
        for i, row in topic_names_tsv.iterrows():
            clarity = row['clarity']
            if clarity != clarity :
                clarity = ''
            mixed = row['mixed']
            if mixed != mixed:
                mixed = ''
            name = row['name']
            if name != name:
                name = ''
            tag = row['tag']
            if tag != tag:
                tag = ''
            topic_names.append( {'name':name, 'clarity':clarity, 'mixed':mixed, 'tag':tag} )

    topic_html_signatures = []
    for i in tqdm(range(n_topics)):
        topic_html_signatures.append(topic_signature_html(m, i, 100, colormap))

    topic_doc_html_signatures = []
    for i in tqdm(range(n_topics)):
        topic_doc_html_signatures.append(topic_document_signature_html(corpus, i, TD, m, doc_list, n_docs_per_topic, colormap))

    @app.route('/topic_names')  # or @route('/scidp')
    def generate_html_page_for_topic_names():

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
                    <form id="eval_form" method="post" action="topic_names">
                    <section>
                    <h2>Instructions</h2>
                    <p>You will be shown topics related to natural language processing
                    and asked to judge them.</p>
                    <p>Each topic is represented by a weighted collection of words, phrases,
                    and entities, where the darker the color, the more important it is to
                    the topic.</p>
                    <p> Phrases may be missing common function words
                    like &lsquo;of&rsquo;, making &lsquo;part of speech&rsquo; show up as
                    &lsquo;part speech&rsquo;. Other phrases may be truncated or split, e.g.,
                    &lsquo;automatic speech recognition&rsquo; will be displayed as
                    &lsquo;automatic speech&rsquo; and &lsquo;speech recognition&rsquo;.</p>
                    <p>You can click on the name of each entity to see its Wikipedia page. (You may need to choose the most relevant sense for ambiguous entities.</p>
                    <p>For each
                    topic you will also see a list of related papers, which you can click on to view
                    in full.</p>
                    <p>For each phrase associated with the topic you can hover your mouse to
                    see how relevant it is to the topic. For each listed document, the
                    percent of the document that is about the topic is displayed after the
                    title.</p>
                    <p>For each topic, you are asked how clear it is to you. A topic may be
                    unclear because it is a mix of distinct ideas or because it is an area of
                    research that is unfamiliar to you.</p>
                    <p>Your Name: <input name="name" style="width: 300px" /></p>
                    <p>Your Email: <input name="email" style="width: 300px" /></p>
                    </section>
                    """

        for tid in range(0, n_topics):
            html_string += '<section>\n'
            html_string += '    <section>\n'
            html_string += '<h2>Topic ' + str(tid) + '</h2>'
            html_string += '<div class="topic">'

            '''
            html_string += '<p>Relevant entities:</p>'
            html_string += '<blockquote><p>', topic_rep[topic]['entity'], '</p></blockquote>'
            html_string += '<p>Relevant pairs of words:</p>'
            html_string += '<blockquote><p>', topic_rep[topic]['bigram'], '</p></blockquote>'
            '''

            html_string += '<p>Relevant words:</p>'
            html_string += topic_html_signatures[tid]

            html_string += '<p>Relevant documents:</p>'
            html_string += topic_doc_html_signatures[tid]

            html_string += '<p>How clear and coherent is the meaning of this topic?</p>'
            html_string += '<p class="rate">'
            for i, label in enumerate(['Very clear', 'Somewhat clear', 'Not very clear', 'Not clear at all'], 1):
                if len(topic_names) == 0:
                    html_string += '<input type="radio" name="' + str(tid) + '_mean" id="' + str(tid) + '_mean-' + \
                                   str(i) + '" value="' + str(i) + '" />'
                    html_string += '<label for="' + str(tid) + '_mean-' + str(i) + '">' + label + '</label>'
                else:
                    html_string += '<input type="radio" name="' + str(tid) + '_mean" id="' + str(tid) + '_mean-' + \
                                   str(i) + '" value="' + str(i) + '"'
                    if topic_names[tid]['clarity'] == i:
                        html_string += 'checked'
                    html_string += '/>'
                    html_string += '<label for="' + str(tid) + '_mean-' + str(i) + '">' + label + '</label>'

            html_string += '</p>'

            html_string += '<p>Does this look like a combination of two or more distinct topics?</p>'
            html_string += '<p class="rate">'
            for i, label in enumerate(['No', 'Yes'], 1):
                if len(topic_names) == 0:
                    html_string += '<input type="radio" name="' + str(tid) + '_comb" id="' + str(tid) + '_comb-' + \
                                   str(i) + '" value="' + str(i) + '" />'
                    html_string += '<label for="' + str(tid) + '_comb-' + str(i) + '">' + label + '</label>'
                else:
                    html_string += '<input type="radio" name="' + str(tid) + '_comb" id="' + str(tid) + '_comb-' + \
                                   str(i) + '" value="' + str(i) + '" '
                    if topic_names[tid]['mixed'] == i:
                        html_string += 'checked'
                    html_string += '/>'
                    html_string += '<label for="' + str(tid) + '_comb-' + str(i) + '">' + label + '</label>'
            html_string += '</p>'
            html_string += '<p>What short name would you give this topic?</p>'
            html_string += '<p class="rate">'

            if len(topic_names) > 0:
                html_string += '<input name="' + str(tid) + '_name" ' \
                                    'id="' + str(tid) + '_name" style="width: 400px" ' \
                                    'value="'+topic_names[tid]['name']+'"/>'
            else:
                html_string += '<input name="'+str(tid)+'_name" id="'+str(tid)+ '_name" style="width: 400px"/>'

            html_string += '</p>'
            html_string += '<p>Tag</p>'
            html_string += '<p class="rate">'
            if len(topic_names) > 0:
                html_string += '<input name="' + str(tid) + '_tag" id="' + str(
                    tid) + '_tag" style="width: 400px" value="'+topic_names[tid]['tag']+'"/>'
            else:
                html_string += '<input name="' + str(tid) + '_tag" id="' + str(
                    tid) + '_tag" style="width: 400px"/>'
            html_string += '</p>'
            html_string += '</section>'

        html_string += """
                    <section>
                    <p><input id="submitButton" type="submit" name="Submit" value="Submit"
                    /></p>
                    </section>
                    </form>
                    </article>
                    </body>
                    </html>
                """

        return html_string

    @app.post('/topic_names')  # or @route('/login', method='POST')
    def save_html_page():

        postdata = str(request.body.read())
        records = postdata.split('&')
        d = {}
        meta_d = {}
        max_d = 0
        for rec in records:
            if len(rec.split('=')) == 2:
                (key, value) = rec.split('=')
            else:
                key = rec.split('=')
                value = ''
            if "_" not in key:
                meta_d[key] = value
            else:
                (id, field) = key.split('_')
                if d.get(id, None) is None:
                    d[id] = {}
                d.get(id)[field] = value
                if int(id) > int(max_d):
                    max_d = id

        rows = []
        key_list = sorted([int(k) for k in d.keys()])
        for i in key_list:
            rows.append(d.get(str(i)))
        df = pd.DataFrame(data=rows)
        df.to_csv(topicmodel_dir + '/topic_names.tsv', sep='\t')

        html = """<html>
                    <head>
                    <title>Topic Evaluation</title></head>
                    <body>
                    <h1>Thank you</h1>
                    </body>
                    </html>
        """

    run(app, host='0.0.0.0', port=port, debug=True)

if __name__ == "__main__":
    main()
