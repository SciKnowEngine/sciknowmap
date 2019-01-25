import bottle

from beaker.middleware import SessionMiddleware
from cork import Cork
import logging

import math
from sciknowmap.mallet import Mallet
from sciknowmap.corpus import Corpus,Document
import os
import warnings
import sys
import codecs
import numpy
import argparse
import json
import pickle
import urllib.parse
import click
from numpy.linalg import norm
import numpy as np
import operator
from tqdm import tqdm
import random
import pandas as pd
import json

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
        pmid = doc_list[doc_id]
        if corpus.get_document(pmid) is not None:
            doc = corpus.get_document(pmid)
        else:
            doc = Document()
            doc.authors = []
            doc.url = ""
            doc.title = ""
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
@click.argument('page_size', type=click.INT)
def main(topicmodel_dir, corpus_dir, port, n_docs_per_topic, page_size):

    logging.basicConfig(format='localhost - - [%(asctime)s] %(message)s', level=logging.DEBUG)
    log = logging.getLogger(__name__)
    #bottle.debug(True)

    # Use users.json and roles.json in the local example_conf directory
    aaa = Cork(topicmodel_dir, email_sender='gully.burns@chanzuckerberg.com', smtp_url='smtp://smtp.gmail.com')

    app = bottle.app()
    session_opts = {
        'session.cookie_expires': True,
        'session.encrypt_key': 'please use a random key and keep it secret!',
        'session.httponly': True,
        'session.timeout': 3600 * 24,  # 1 day
        'session.type': 'cookie',
        'session.validate_key': True,
    }
    app = SessionMiddleware(app, session_opts)

    #
    # Loads the topic model
    #
    MALLET_PATH = '/Users/gullyburns/Applications/mallet-2.0.8/bin'

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
    for i in range(n_topics):
        r = lambda: random.randint(0,255)
        colors.append('#%02X%02X%02X' % (r(),r(),r()))

    colormap = np.array(colors)
    print(len(colormap))

    # Load the names
    curation_data_path = topicmodel_dir + "/curation_data.json"
    if os.path.isfile(curation_data_path) :
        with open(curation_data_path) as f:
            curation_data = json.load(f)
    else:
        curation_data = {}

    topic_html_signatures = []
    for i in tqdm(range(n_topics)):
        topic_html_signatures.append(topic_signature_html(m, i, 100, colormap))

    topic_doc_html_signatures = []
    for i in tqdm(range(n_topics)):
        topic_doc_html_signatures.append(topic_document_signature_html(corpus, i, TD, m, doc_list, n_docs_per_topic, colormap))

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # LOGIN STUFF
    # Copyright (C) 2013 Federico Ceratto and others, see AUTHORS file.
    # Released under LGPLv3+ license, see LICENSE.txt
    #
    # Cork example web application
    #
    # The following users are already available:
    #  admin/admin, demo/demo

    # #  Bottle methods  # #

    def postd():
        return bottle.request.forms

    def post_get(name, default=''):
        return bottle.request.POST.get(name, default).strip()

    @bottle.post('/login')
    def login():
        """Authenticate users"""
        username = post_get('username')
        password = post_get('password')
        aaa.login(username, password, success_redirect='/', fail_redirect='/login')

    @bottle.route('/user_is_anonymous')
    def user_is_anonymous():
        if aaa.user_is_anonymous:
            return 'True'

        return 'False'

    @bottle.route('/logout')
    def logout():
        aaa.logout(success_redirect='/login')

    @bottle.post('/register')
    def register():
        """Send out registration email"""
        aaa.register(post_get('username'), post_get('password'), post_get('email_address'))
        return 'Please check your mailbox.'

    @bottle.route('/validate_registration/:registration_code')
    def validate_registration(registration_code):
        """Validate registration, create user account"""
        aaa.validate_registration(registration_code)
        return 'Thanks. <a href="/login">Go to login</a>'

    @bottle.post('/reset_password')
    def send_password_reset_email():
        """Send out password reset email"""
        aaa.send_password_reset_email(
            username=post_get('username'),
            email_addr=post_get('email_address')
        )
        return 'Please check your mailbox.'

    @bottle.route('/change_password/:reset_code')
    @bottle.view('password_change_form')
    def change_password(reset_code):
        """Show password change form"""
        return dict(reset_code=reset_code)

    @bottle.post('/change_password')
    def change_password():
        """Change password"""
        aaa.reset_password(post_get('reset_code'), post_get('password'))
        return 'Thanks. <a href="/login">Go to login</a>'

    @bottle.route('/restricted_download')
    def restricted_download():
        """Only authenticated users can download this file"""
        aaa.require(fail_redirect='/login')
        return bottle.static_file('static_file', root='.')

    @bottle.route('/my_role')
    def show_current_user_role():
        """Show current user role"""
        session = bottle.request.environ.get('beaker.session')
        print
        "Session from simple_webapp", repr(session)
        aaa.require(fail_redirect='/login')
        return aaa.current_user.role

    # Admin-only pages

    @bottle.route('/admin')
    @bottle.view('admin_page')
    def admin():
        """Only admin users can see this"""
        aaa.require(role='admin', fail_redirect='/sorry_page')
        return dict(
            current_user=aaa.current_user,
            users=aaa.list_users(),
            roles=aaa.list_roles()
        )

    @bottle.post('/create_user')
    def create_user():
        try:
            aaa.create_user(postd().username, postd().role, postd().password)
            return dict(ok=True, msg='')
        except Exception as e:
            return dict(ok=False, msg=e.message)

    @bottle.post('/delete_user')
    def delete_user():
        try:
            aaa.delete_user(post_get('username'))
            return dict(ok=True, msg='')
        except Exception as e:
            print
            repr(e)
            return dict(ok=False, msg=e.message)

    @bottle.post('/create_role')
    def create_role():
        try:
            aaa.create_role(post_get('role'), post_get('level'))
            return dict(ok=True, msg='')
        except Exception as e:
            return dict(ok=False, msg=e.message)

    @bottle.post('/delete_role')
    def delete_role():
        try:
            aaa.delete_role(post_get('role'))
            return dict(ok=True, msg='')
        except Exception as e:
            return dict(ok=False, msg=e.message)

    # Static pages

    @bottle.route('/login')
    @bottle.view('login_form')
    def login_form():
        """Serve login form"""
        return {}

    @bottle.route('/sorry_page')
    def sorry_page():
        """Serve sorry page"""
        return '<p>Sorry, you are not authorized to perform this action</p>'

    @bottle.route('/')
    def generate_top_page():
        """Only authenticated users can see this"""
        aaa.require(fail_redirect='/login')

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
                           """
        html_string += '<body>'+ aaa.current_user.username +' <a href="/admin">[admin]</a><a href="/logout">[logout]</a>'
        html_string += """
                           <h1>Topic Evaluation</h1>
                           <article>
                           <form id="eval_form" method="post" action="topics">
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
                           </section>
                           """


        html_string += '<fieldset id="page">\n'
        for tid in range(0, n_topics, page_size):
            html_string += '<input type="radio" name="page" value="'+str(tid)+'-'+str(tid+page_size-1)+'">['+str(tid)+'-'+str(tid+page_size-1)+']</input>'
        html_string += '</fieldset>\n'

        html_string += """
                    <p><input id="submitButton" type="submit" name="Curate" value="Curate"
                    /></p>
                    </section>
                    </form>
                    </article>
                    </body>
                    </html>
                """

        return html_string


    @bottle.post('/topics')  # or @route('/login', method='POST')
    def generate_html_page_for_topic_names():

        formdata = urllib.parse.unquote(str(bottle.request.body.read()))
        form = {}
        formrows = formdata.split('&')
        for row in formrows:
            if len(row.split('=')) == 2:
                (key, value) = row.split('=')
            else:
                key = row.split('=')
                value = ''
            form[key] = value

        (start_end) = form["b'page"]
        (start, end) = start_end.split('-')

        topic_names = []
        user = aaa.current_user.username
        for i in range(0, n_topics):
            if curation_data.get(user,None) is not None:
                existing_curation_data = curation_data[user]
                if existing_curation_data.get(str(i), None) is not None:
                    row = existing_curation_data[str(i)]
                    clarity = row.get('comb','')
                    mixed = row.get('mean','')
                    name = row.get('name','')
                    tag = row.get('tag','')
                    topic_names.append({'name': name, 'clarity': clarity, 'mixed': mixed, 'tag': tag})
                else:
                    topic_names.append({'name': '', 'clarity': '', 'mixed': '', 'tag': ''})
            else:
                topic_names.append({'name': '', 'clarity': '', 'mixed': '', 'tag': ''})

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
                    """

        html_string += "<h1>Welcome " + aaa.current_user.username + ". <br>Curation of Topics ["+start_end+"]</h1>"
        html_string += '<form id="curation_form" method="post" action="save_topics">'

        for tid in range(int(start), int(end)):
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
                    if topic_names[tid]['clarity'] == str(i):
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
                    if topic_names[tid]['mixed'] == str(i):
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

    @bottle.post('/save_topics')  # or @route('/login', method='POST')
    def save_html_page():

        postdata = str(bottle.request.body.read())
        records = postdata.split('&')
        user = aaa.current_user.username
        if curation_data.get(user, None) is None:
            curation_data[user] = {}

        d = curation_data[user]
        meta_d = {}
        max_d = 0
        min_d = 10000
        for rec in records:
            if rec[:2] == 'b\'':
                rec = rec[2:]
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
                if int(id) < int(min_d):
                    min_d = id

        with open(curation_data_path, 'w') as f:
            json.dump(curation_data, f)

        #rows = []
        #key_list = sorted([int(k) for k in d.keys()])
        #for i in key_list:
        #    rows.append(d.get(str(i)))
        #df = pd.DataFrame(data=rows)
        #df.to_csv(topicmodel_dir + '/topic_names_'+'_'+user+'_'+min_d+'_'+max_d+'.tsv', sep='\t')

        return '<html><head><title>Topic Evaluation</title><meta http-equiv = "refresh" content = "1; url = ' + \
                'https://localhost:' + str(port) + '" /></head><body><h1>Thank you</h1></body></html>'

    bottle.run(app, host='0.0.0.0', port=port, debug=True)

if __name__ == "__main__":
    main()
