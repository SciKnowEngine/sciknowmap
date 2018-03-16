#!/usr/bin/env python3

# Topic Mapping Script
# Gully Burns

import math
import operator
import os
import pickle
import random
from datetime import datetime
from matplotlib import pyplot as plt
from colour import Color
from urllib.parse import unquote_plus

import click
import numpy as np
import pandas as pd
from numpy.linalg import norm, eigh
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm

from utils.mallet import Mallet
from utils.corpus import Corpus

def compute_luminance(c):
    (r,g,b) = tuple(int(c[i:i + 2], 16) for i in (1, 3, 5))
    return (.299 * r) + (.587 * g) + (.114 * b)

def all_topics_signature_html(DT, m, n_words, colormap):
    html_signature = '<p>'
    html_signature += '</br>'.join([topic_signature_html(m, (i,1.0), n_words, colormap) for i in range(DT.shape[1])])
    html_signature += '</p>'

    return html_signature

#
# Provides HTML code for a single topic signature based on greyscale coding
# for each word
#
def topic_signature_html(m, t_tuple, n_words, colormap, topic_name=None, global_min=None, global_max=None):
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

    ret = '<emph><font color="' + color + '">&#x25A0; </font>#' + str(t_id)
    if topic_name is not None:
        ret += ' ' + topic_name + ' '

    ret += ' (' + t_percent_2sf + '): </emph>'

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

def document_signature_html(corpus, doc_id, DT, m, doc_list, n_topics, n_words, colormap, topic_names=None):
    doc_count = DT.shape[0]
    top_topics = sorted(
        enumerate(DT[doc_id]), reverse=True, key=operator.itemgetter(1)
    )[:n_topics]

    doc = corpus[doc_list[doc_id]]
    html_signature = '<p><b>' + doc.title + '</b></br>'
    html_signature += '<i>' + ', '.join(doc.authors) + '</i>'
    html_signature += '</br>'
    if topic_names is None:
        html_signature += '</br>'.join([topic_signature_html(m, top_topics[i], n_words, colormap, None) for i in range(n_topics)])
    else:
        html_signature += '</br>'.join([topic_signature_html(m, top_topics[i], n_words, colormap, topic_names[top_topics[i][0]]) for i in range(n_topics)])

    html_signature += '</p>'

    return html_signature
#
# SCRIPT TO RUN TOPIC MAPPING VISUALIZATION UNDER DIFFERENT METHODS
#
@click.command()
@click.argument('topicmodel_dir', type=click.STRING)
@click.argument('corpus_dir', type=click.Path(exists=True))
@click.argument('viz_dir', type=click.Path())
@click.argument('title', type=click.STRING)
@click.option('--no_bad_topics', 'mode', flag_value='no_bad_topics')
def main(topicmodel_dir, corpus_dir, viz_dir, title, mode):

    MALLET_PATH = '/usr/local/bin/mallet'

    if os.path.exists(viz_dir) is False:
        os.makedirs(viz_dir)

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

    #
    # Build color maps from previous work
    #
    color_file_path = topicmodel_dir + "/topic_colors.tsv"
    colors = []
    if os.path.exists(color_file_path):
        colors_tsv = pd.read_csv(color_file_path, sep='\t')
        for i, row in colors_tsv.iterrows():
            c = row['colors']
            if c != c:
                c = '#000000'
            colors.append(c)
    else:
        for i in range(n_topics):
            r = lambda: random.randint(0, 255)
            colors.append('#%02X%02X%02X' % (r(), r(), r()))
        df = pd.DataFrame({'colors': colors})
        df.to_csv(topicmodel_dir + '/topic_colors.tsv', sep='\t')

    colormap = np.array(colors)

    if mode == 'no_bad_topics' :

        #
        # Load / Build Topic Labels for Map.
        #
        topic_names_file_path = topicmodel_dir + "/topic_names.tsv"
        topic_name_list = []
        topic_score_list = []
        if os.path.exists(topic_names_file_path):
            colors_tsv = pd.read_csv(topic_names_file_path, sep='\t')
            for i, row in colors_tsv.iterrows():
                name = row['name']
                if name != name:
                    name = ''
                score_text = row['mean']
                if score_text != score_text:
                    score = 4
                else:
                    score = int(score_text)
                topic_name_list.append(unquote_plus(name))
                topic_score_list.append(score)
        else:
            for t_id in range(n_topics):
                word_weights = sorted(
                    m.topics[t_id].items(), key=operator.itemgetter(1), reverse=True
                )[:5]
                # use the 5 top words in the topic.
                topic_name_list.append(" ".join([ww[0] for ww in word_weights]))
                topic_score_list.append(1)

        topic_scores = np.array(topic_score_list)
        topic_names = np.array(topic_name_list)

        good_topics = np.where(topic_scores < 3)
        bad_topics = np.where(topic_scores > 2)
        colormap[bad_topics] = Color("grey").get_hex()
        topic_names[bad_topics] = ''

        dt_filtered = DT.transpose()[good_topics].transpose()
        dt_normalized = normalize(dt_filtered, axis=1, norm='l1')
        filtered_topic_names = np.array(topic_names)[np.where(topic_scores<3)].tolist()
        DT = dt_normalized
        topic_names = filtered_topic_names
        n_topics = len(topic_names)

    if os.path.exists(viz_dir) is False:
        os.mkdirs(viz_dir)

    tsne_lda_pkl_path = viz_dir + "/tsne_lda.pkl"

    if os.path.isfile(tsne_lda_pkl_path) is False:

        tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.7, method='exact', init='pca')
        tsne_lda = tsne_model.fit_transform(DT)

        # save the t-SNE model
        tsne_lda_pkl_file = open(tsne_lda_pkl_path, 'wb')
        pickle.dump(tsne_lda, tsne_lda_pkl_file)
        tsne_lda_pkl_file.close()

    else:

        tsne_lda_pkl_file = open(tsne_lda_pkl_path, 'rb')
        tsne_lda = pickle.load(tsne_lda_pkl_file)
        tsne_lda_pkl_file.close()

    top_topics_list = []
    for i in range(n_docs):
        tuple = sorted(enumerate(DT[i]), reverse=True, key=operator.itemgetter(1))[:1]
        top_topics_list.append(tuple[0][0])
    top_topics = np.asarray(top_topics_list)

    # compute the densest positions for each individual topics.
    topic_maxima_list = []
    params = {'bandwidth': np.logspace(-1, 1, 5)}
    grid = GridSearchCV(KernelDensity(), params)
    print("Computing topic maps density distributions")
    for i in tqdm(range(n_topics)):
        X = tsne_lda[np.where(top_topics == i)]
        grid.fit(X)
        kde = grid.best_estimator_
        densities = kde.score_samples(X)
        local_maxima = X[np.where(densities == np.max(densities))]
        topic_maxima_list.append(local_maxima[0])
    topic_maxima = np.asarray(topic_maxima_list)

    topic_keys = []
    for i in range(DT.shape[0]):
        topic_keys += DT[i].argmax(),

    color = []
    for i in tqdm(range(DT.shape[0])):
        color.append(colormap[topic_keys][i])

    # plot the result
    vis_x = tsne_lda[:, 0]
    vis_y = tsne_lda[:, 1]

    now = datetime.now().strftime("%d-%b-%Y-%H%M%S")
    plt.figure(figsize=(8, 8), dpi=300)
    plt.scatter(vis_x, vis_y, c=color, cmap=plt.cm.get_cmap("jet", 10), s=0.6, alpha=0.8, marker="o", edgecolors='none')
    plt.scatter(topic_maxima[:, 0], topic_maxima[:, 1], c=colormap, cmap=plt.cm.get_cmap("jet", 10), s=10, alpha=0.8, marker="x", linewidths=0.1)
    plt.clim(-0.5, 9.5)
    plt.savefig(viz_dir + '/scatterplot_' + now + '_tsne.png')

if __name__ == '__main__':
    main()
