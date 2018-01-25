#!/usr/bin/env python3

# Topic Mapping Script
# Gully Burns

import math
import operator
import os
import pickle
import random
from datetime import datetime
from urllib.parse import unquote_plus

import bokeh.plotting as bp
import click
import numpy as np
import pandas as pd
from bokeh.io import output_file, show
from bokeh.models import ColumnDataSource, HoverTool, TapTool, OpenURL, LabelSet, Label
from bokeh.models import PanTool, BoxZoomTool, WheelZoomTool, ResetTool
from numpy.linalg import norm, eigh
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
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
@click.option('--normal', 'mode', flag_value='normal',
              default=True)
@click.option('--clusters', 'mode', flag_value='clusters')
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

    if os.path.exists(viz_dir) is False:
        os.mkdirs(viz_dir)

    tsne_lda_pkl_path = viz_dir + "/tsne_lda.pkl"

    if os.path.isfile(tsne_lda_pkl_path) is False:

        tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99, init='pca')
        tsne_lda = tsne_model.fit_transform(DT)

        # save the t-SNE model
        tsne_lda_pkl_file = open(tsne_lda_pkl_path, 'wb')
        pickle.dump(tsne_lda, tsne_lda_pkl_file)
        tsne_lda_pkl_file.close()

    else:

        tsne_lda_pkl_file = open(tsne_lda_pkl_path, 'rb')
        tsne_lda = pickle.load(tsne_lda_pkl_file)
        tsne_lda_pkl_file.close()

    #
    # Compute a 200 cluster analysis over the XY coordinates
    # 'ells' contains ellipses for each cluster
    #
    gmm_lda_pkl_path = viz_dir + "/cluster_data.pkl"
    if os.path.isfile(gmm_lda_pkl_path) is False:
        gmm = GaussianMixture(n_components=200, covariance_type='full').fit(tsne_lda)
        gmm_lda_pkl_file = open(gmm_lda_pkl_path, 'wb')
        pickle.dump(gmm, gmm_lda_pkl_file)
        gmm_lda_pkl_file.close()
    else:
        gmm_lda_pkl_file = open(gmm_lda_pkl_path, 'rb')
        gmm = pickle.load(gmm_lda_pkl_file)
        gmm_lda_pkl_file.close()

    ells_tuples = []
    clusters = gmm.predict(tsne_lda)
    cluster_topic = []
    for i, (mean, covar) in enumerate(zip(gmm.means_, gmm.covariances_)):
        mean_topic_signature = np.mean(DT[np.where(clusters == i)],axis=0)
        cluster_topic.append(np.argmax(mean_topic_signature))
        v, w = eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / norm(w[0])
        angle = np.arctan(u[1] / u[0])
        ells_tuples.append((mean[0],mean[1],v[0],v[1],angle))
    ells = np.array(ells_tuples)

    # Compute centroids on figure
    top_topics_list = []
    for i in range(n_docs):
        tuple = sorted(enumerate(DT[i]), reverse=True, key=operator.itemgetter(1))[:1]
        top_topics_list.append(tuple[0][0])
    top_topics = np.asarray(top_topics_list)
    topic_centroid_list = []
    for i in range(n_topics):
        centroid = np.mean(tsne_lda[np.where(top_topics == i)],axis=0)
        topic_centroid_list.append(centroid)
    topic_centroids = np.asarray(topic_centroid_list)

    color_keys = []
    if (mode == 'clusters'):
        color_keys = clusters
    else:
        for i in range(DT.shape[0]):
            color_keys += DT[i].argmax(),

    # Code to create the HTML display
    colors = []
    for i in range(200):
        r = lambda: random.randint(0,255)
        colors.append('#%02X%02X%02X' % (r(),r(),r()))

    colormap = np.array(colors)
    luminance = [compute_luminance(c) for c in colors]
    text_color = [('black' if l > 125 else 'white') for l in luminance]

    #
    # Load / Build Topic Labels for Map.
    #
    topic_names_file_path = topicmodel_dir + "/topic_names.tsv"
    topic_names = []
    if os.path.exists(topic_names_file_path):
        topic_names_tsv = pd.read_csv(topic_names_file_path, sep='\t')
        for i, row in topic_names_tsv.iterrows():
            name = row['name']
            if name != name:
                name = ''
            topic_names.append(unquote_plus(name))
    else:
        for i in range(n_topics):
            # use the 5 top words in the topic.
            topic_names.append(topic_signature_html(m, i, 5, colormap))

    print( "Generating Document Signature Data")
    html_signatures = []
    for i in tqdm(range(n_docs)):
        html_signatures.append(document_signature_html(corpus, i, DT, m, doc_list, 5, 10, colormap, topic_names))

    #display(HTML(html_signatures[0]))

    doc_count = DT.shape[0]
    doc_urls = [corpus[doc_list[i]].url for i in range(doc_count)]

    markers = []
    for i in range(DT.shape[0]):
        if 'gbook' in doc_list[i]:
            markers.append('triangle')
        else:
            markers.append('circle')

    num_example = len(DT)

    hover = HoverTool(tooltips="""
        <div>
            <span>
                @html_signatures{safe}
            </span>
        </div>
        """
        )

    pan = PanTool()
    boxzoom = BoxZoomTool()
    wheelzoom = WheelZoomTool()
    resetzoom = ResetTool()
    tap = TapTool(callback=OpenURL(url="@doc_urls"))

    cds = ColumnDataSource({
        "x": tsne_lda[:, 0],
        "y": tsne_lda[:, 1],
        "color": colormap[color_keys],
        "html_signatures": html_signatures,
        "doc_urls": doc_urls,
        "marker": markers
    })

    #topic_labels = ['&#x25A0; ' + topic_names[i] for i in range(n_topics)]
    label_cds = ColumnDataSource(data=dict(
            x=topic_centroids[:,0],
            y=topic_centroids[:,1],
            label=topic_names,
            text_color=text_color,
            color=colormap,
            text_size=[9] * 200))

    # plot_lda = bp.figure(plot_width=1400, plot_height=1100,
    #                     title=title,
    #                     tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",
    #                     x_axis_type=None, y_axis_type=None, min_border=1)

    plot_lda = bp.figure(plot_width=1400, plot_height=1100,
                         title=title,
                         tools=[pan, boxzoom, wheelzoom, resetzoom, hover, tap],
                         active_drag=pan,
                         active_scroll=wheelzoom,
                         x_axis_type=None, y_axis_type=None, min_border=1)

    # HACK TO GENERATE DIFFERENT PLOTS FOR CIRCLES AND TRIANGLES
    '''
    marker_types = ['circle', 'triangle']
    for mt in marker_types:
        x = []
        y = []
        color = []
        html_sig = []
        doc_url = []
        print(mt)
        for i in tqdm(range(DT.shape[0])):
            if markers[i] == mt:
                x.append(tsne_lda[i, 0])
                y.append(tsne_lda[i, 1])
                color.append(colormap[topic_keys][i])
                html_sig.append(html_signatures[i])
                doc_url.append(doc_urls[i])
        cds_temp = ColumnDataSource({
            "x": x,
            "y": y,
            "color": color,
            "html_signatures": html_sig,
            "doc_urls": doc_url
        })

        plot_lda.scatter('x', 'y', color='color', marker=mt, source=cds_temp)
    '''
    plot_lda.scatter('x', 'y', color='color', radius=0.02, alpha=0.7, source=cds)

    if (mode == 'clusters'):
        plot_lda.ellipse(x = ells[...,0],
                        y = ells[...,1],
                        width = ells[...,2],
                        height = ells[...,3],
                        angle = ells[...,4],
                        alpha=1,
                        line_width=1,
                        fill_color=None,
                        line_color="black")

    labels = LabelSet(x='x', y='y', text='label', source=label_cds,
                      text_align='center', text_color= 'color', text_font_size="6pt")

    '''
    labels = LabelSet(x='x', y='y', text='label', background_fill_color='color', source=label_cds,
                      text_align='center', text_color= 'text_color', text_font_size="6pt",
                      background_fill_alpha=0.7)
    '''
    plot_lda.add_layout(labels)

    now = datetime.now().strftime("%d-%b-%Y-%H%M%S")

    output_file(viz_dir + '/scatterplot' + now + '.html', title=title, mode='cdn',
                root_dir=None)
    show(plot_lda)

    html_string = """
        <html>
        <head>
        <title>Topic Legend</title>
        </head>
        <body>
        """

    html_string += all_topics_signature_html(DT, m, 10, colormap)
    html_string + "<\body></html>"
    output = open(viz_dir + '/legend' + now + '.html', 'w')
    output.write(html_string)
    output.close()

if __name__ == '__main__':
    main()
