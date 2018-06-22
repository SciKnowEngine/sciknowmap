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
from colour import Color
import json
import re

import bokeh.plotting as bp
import click
import numpy as np
import pandas as pd
from bokeh.io import output_file, show
from bokeh.models import ColumnDataSource, HoverTool, TapTool, OpenURL, LabelSet, Label
from bokeh.models import PanTool, BoxZoomTool, WheelZoomTool, ResetTool
from numpy.linalg import norm, eigh
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
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
    html_signature += '</br>'.join([topic_signature_html(m, i, 1.0, n_words, colormap) for i in range(DT.shape[1])])
    html_signature += '</p>'

    return html_signature

#
# Provides HTML code for a single topic signature based on greyscale coding
# for each word
#
def topic_signature_html(topic_word_tuples, t_id, t_percent, n_words, colormap, topic_name=None):
    ret = topic_signature_head_html(t_id, t_percent,colormap, topic_name)
    ret += topic_signature_body_html(topic_word_tuples, t_id, n_words)
    return ret

def topic_signature_head_html(t_id, t_percent, colormap, topic_name=None):
    t_percent_2sf = '%s' % float('%.2g' % t_percent)
    color = colormap[t_id]
    ret = '<emph><font color="' + color + '">&#x25A0; </font>#' + str(t_id)
    if topic_name is not None:
        ret += ' ' + topic_name + ' '

    ret += ' (' + t_percent_2sf + '): </emph>'
    return ret

def topic_signature_body_html(topic_word_tuples, t_id, n_words, global_min=None, global_max=None):

    ret = ""
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
        topic_word_tuples[t_id].items(), key=operator.itemgetter(1), reverse=True
    )[:n_words]

    vals = [x[1] for x in word_weights]
    val_max = max(vals)
    val_min = math.sqrt(min(vals) / 2)
    val_diff = float(val_max - val_min)
    if global_min and global_max:
        global_diff = float(global_max - global_min)

    for (y, z) in sorted(word_weights, key=lambda x: x[1],
                         reverse=True):

        p = float(z - val_min) / val_diff

        if global_min and global_max:
            q = float(z - global_min) / global_diff
        else:
            q = p

        ret += '<span style="color:%s">%s</span>\n' % (float_to_greyscale(p), y.replace('_', '&nbsp;'))

    return ret


def document_signature_html(corpus, doc_id, DT, topic_word_tuples, doc_list, n_topics, n_words, colormap, topic_names=None):
    doc_count = DT.shape[0]
    top_topics = sorted(
        enumerate(DT[doc_id]), reverse=True, key=operator.itemgetter(1)
    )[:n_topics]

    doc = corpus[doc_list[doc_id]]
    html_signature = '<p><b>' + doc.title + '</b></br>'
    html_signature += '<i>' + ', '.join(doc.authors) + '</i>'
    html_signature += '</br>'

    html_signature += '</br>'.join(
            [topic_signature_head_html(top_topics[i][0], top_topics[i][1], colormap, topic_names[top_topics[i][0]])
             + "|||+all_topic_html["+str(top_topics[i][0])+"]+|||" for i in range(n_topics)])

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
@click.option('--run_code', default="0", help='Run number.')
@click.option('--n_components', default=2, help='TSNE Number of components.')
@click.option('--perplexity', default=35.0, help='TSNE Perplexity.')
@click.option('--method', default="barnes_hut", help='TSNE Method.')
@click.option('--angle', default=0.5, help='TSNE Angle.')
@click.option('--no_bad_topics', is_flag=True)
@click.option('--merge_topics', is_flag=True)
@click.option('--lightweight_test', is_flag=True)
def main(topicmodel_dir, corpus_dir, viz_dir, title, run_code, n_components, perplexity, method, angle, no_bad_topics, merge_topics, lightweight_test):

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

    topic_word_tuples = m.topics

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
    color_threshold = 175
    text_color = [('black' if compute_luminance(c) > color_threshold else c) for c in colors]
    background_alpha = [(1.0 if compute_luminance(c) > color_threshold else 0.6) for c in colors]
    background_color = [(c if compute_luminance(c) > color_threshold else "#ffffff") for c in colors]

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
                topic_word_tuples[t_id].items(), key=operator.itemgetter(1), reverse=True
            )[:5]
            # use the 5 top words in the topic.
            topic_name_list.append(" ".join([ww[0] for ww in word_weights]))
            topic_score_list.append(1)

    topic_scores = np.array(topic_score_list)
    topic_names = np.array(topic_name_list)

    good_topics = np.where(topic_scores<3)
    bad_topics_flag = np.where(topic_scores>2)

    if no_bad_topics:
        dt_filtered = DT.transpose()[good_topics].transpose()
        dt_normalized = normalize(dt_filtered, axis=1, norm='l1')
        filtered_topic_names = np.array(topic_names)[np.where(topic_scores<3)].tolist()
        filtered_colormap = np.array(colormap)[np.where(topic_scores < 3)]
        filtered_colors = filtered_colormap.tolist()
        filtered_text_color = [('black' if compute_luminance(c) > color_threshold else c) for c in filtered_colors]
        filtered_background_alpha = [(1.0 if compute_luminance(c) > color_threshold else 0.6) for c in filtered_colors]
        filtered_background_color = [(c if compute_luminance(c) > color_threshold else "#ffffff") for c in filtered_colors]
        filtered_topic_word_tuples = []
        for i in good_topics[0]:
            filtered_topic_word_tuples.append(topic_word_tuples[i])

        DT = dt_normalized
        topic_names = filtered_topic_names
        colormap = filtered_colormap
        n_topics = len(topic_names)
        text_color = filtered_text_color
        background_alpha = filtered_background_alpha
        background_color = filtered_background_color
        topic_word_tuples = filtered_topic_word_tuples

    #
    # merging topics with identical names if required
    #
    if merge_topics:
        to_merge = {}
        topics_to_keep = []
        for i,tn in enumerate(topic_names):
            if( to_merge.get(tn, None) is None ) :
                tids = []
                tids.append(i)
                topics_to_keep.append(i)
                to_merge[tn] = tids
            else:
                tids = to_merge.get(tn)
                tids.append(i)

        for tn in to_merge.keys():
            topics_list = to_merge.get(tn)
            topics_array = np.asarray(topics_list)
            if len(topics_list) > 1:
                dt_extracted = DT.transpose()[topics_array].transpose()
                dt_averaged = np.mean(dt_extracted, axis=1)
                DT[:,topics_array[0]] = dt_averaged

                #
                # Do we want any statistics about merged topics?
                #

                merged_topic_word_tuple = {}
                for word_tuple_id in topics_list:
                    word_tuple = topic_word_tuples[word_tuple_id]
                    for word in word_tuple.keys():
                        word_count = word_tuple[word]
                        if merged_topic_word_tuple.get(word, None) is None :
                            merged_topic_word_tuple[word] = word_count
                        else :
                            merged_topic_word_tuple[word] = merged_topic_word_tuple[word] + word_count
                topic_word_tuples[topics_array[0]] = merged_topic_word_tuple

        merged_topics = np.asarray(topics_to_keep)
        dt_filtered = DT.transpose()[merged_topics].transpose()
        dt_normalized = normalize(dt_filtered, axis=1, norm='l1')
        filtered_topic_names = np.array(topic_names)[merged_topics].tolist()
        filtered_colormap = np.array(colormap)[merged_topics]
        filtered_colors = filtered_colormap.tolist()
        filtered_text_color = [('black' if compute_luminance(c) > color_threshold else c) for c in filtered_colors]
        filtered_background_alpha = [(1.0 if compute_luminance(c) > color_threshold else 0.6) for c in filtered_colors]
        filtered_background_color = [(c if compute_luminance(c) > color_threshold else "#ffffff") for c in filtered_colors]
        filtered_topic_word_tuples = []
        for i in topics_to_keep:
            filtered_topic_word_tuples.append(topic_word_tuples[i])

        DT = dt_normalized
        topic_names = filtered_topic_names
        colormap = filtered_colormap
        n_topics = len(topic_names)
        text_color = filtered_text_color
        background_alpha = filtered_background_alpha
        background_color = filtered_background_color
        topic_word_tuples = filtered_topic_word_tuples

    if os.path.exists(viz_dir) is False:
        os.mkdirs(viz_dir)

    run_signature = "dim"+str(n_components)+\
                        "__"+str(run_code)+\
                        "__ang"+str(angle)+\
                        "__"+method+\
                        "__perp"+str(perplexity)+\
                        "__nbt"+str(no_bad_topics)+\
                        "__mt"+str(merge_topics)

    tsne_lda_pkl_path = viz_dir+"/tsne_data__" + run_signature + ".pkl"

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

    color_keys = []
    for i in range(DT.shape[0]):
        color_keys += DT[i].argmax(),

    topic_keys = []
    for i in range(DT.shape[0]):
        topic_keys += DT[i].argmax(),

    print( "Generating Document Signature Data")
    html_signatures = []
    for i in tqdm(range(n_docs)):
        html_signatures.append(document_signature_html(corpus, i, DT, topic_word_tuples, doc_list, 5, 10, colormap, topic_names))

    topic_signatures = []
    for i in tqdm(range(n_topics)):
        topic_signatures.append(topic_signature_body_html(topic_word_tuples, i, 10))
    all_topic_html = json.dumps(topic_signatures)
    #display(HTML(html_signatures[0]))

    doc_count = DT.shape[0]
    #doc_urls = [corpus[doc_list[i]].url for i in range(doc_count)]
    #doc_urls = ["http://bigdatau.org/resource/" + corpus[doc_list[i]].id for i in range(doc_count)]

    doc_urls = []
    markers = []
    for i in range(DT.shape[0]):
        if 'gbook' in doc_list[i]:
            markers.append('triangle')
            doc_urls.append(corpus[doc_list[i]].url)
        else:
            markers.append('circle')
            doc_urls.append("http://bigdatau.org/resource/"+corpus[doc_list[i]].id)

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

    label_cds = ColumnDataSource(data=dict(
            x=topic_maxima[:,0],
            y=topic_maxima[:,1],
            label=topic_names,
            text_color=text_color,
            background_alpha=background_alpha,
            background_color=background_color,
            text_size=[9] * 200))

    plot_lda = bp.figure(plot_width=1400, plot_height=1100,
                         title=title,
                         tools=[pan, boxzoom, wheelzoom, resetzoom, hover, tap],
                         active_drag=pan,
                         active_scroll=wheelzoom,
                         x_axis_type=None, y_axis_type=None, min_border=1)

    # HACK TO GENERATE DIFFERENT PLOTS FOR CIRCLES AND TRIANGLES

    marker_types = ['circle', 'triangle']
    #marker_types = ['circle']
    number_of_points = DT.shape[0]
    if lightweight_test:
        number_of_points = 100
    for mt in marker_types:
        x = []
        y = []
        color = []
        html_sig = []
        doc_url = []
        print(mt)
        for i in tqdm(range(number_of_points)):
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
        plot_lda.scatter('x', 'y', color='color', marker=mt, alpha=0.7, source=cds_temp)

    labels = LabelSet(x='x', y='y', text='label', background_fill_color='background_color', source=label_cds,
                      text_align='center', text_color= 'text_color', text_font_size="8pt",
                      background_fill_alpha='background_alpha')

    plot_lda.add_layout(labels)

    now = datetime.now().strftime("%d-%b-%Y-%H%M%S")

    temp_file = viz_dir + '/temp' + now + '.html'
    scatterplot_file = viz_dir + '/scatterplot' + now + '.html'
    output_file(temp_file, title=title, mode='cdn', root_dir=None)
    show(plot_lda)

    # Some sneakiness for you...
    # Load the generated HTML and slip in extra code.
    with open(temp_file, "r") as r, open(scatterplot_file, "w") as w:
        for line in r:
            match = re.search("var docs_json", line)
            if match:
                new_line = "var all_topic_html = " + all_topic_html + "\n"
                line = new_line + line
            line = re.sub("\|\|\|", '"', line)
            w.write(line)


    html_string = """
        <html>
        <head>
        <title>Topic Legend</title>
        </head>
        <body>
        """
    html_string += all_topics_signature_html(DT, topic_word_tuples, 10, colormap)
    html_string + "<\body></html>"

    output = open(viz_dir + '/legend' + now + '.html', 'w')
    output.write(html_string)
    output.close()

if __name__ == '__main__':
    main()
