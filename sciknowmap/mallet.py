# Mallet
# Jonathan Gordon

import sys
import os
import tempfile
import random
import re
import subprocess
import multiprocessing as mp
import operator

from numpy import zeros
from itertools import combinations

from sciknowmap.lx import StopLexicon
from collections import defaultdict
from elasticsearch import Elasticsearch, helpers
import time


# Parameters

PROCESSES = int(.5 * mp.cpu_count())
#PROCESSES = 1

OPTIMIZE_INTERVAL = 10

class Mallet:

    def __init__(self, path, corpus=None, num_topics=200, bigrams=False,
                 iters=1000, prefix=None):
        self.path = path

        if prefix:
            self.prefix = prefix
        else:
            rand_prefix = hex(random.randint(0, 0xffffff))[2:] + '-'
            self.prefix = os.path.join(tempfile.gettempdir(), rand_prefix)

        if os.path.isdir(self.prefix) is False:
            os.makedirs(self.prefix)

        self.dtfile = self.prefix + '/composition.txt'
        self.wtfile = self.prefix + '/word-topic-counts.txt'
        self.omfile = self.prefix + '/model.mallet'
        self.inffile = self.prefix + '/inferencer'
        self.tkfile = self.prefix + '/keys.txt'
        self.wtkfile = self.prefix + '/weighted-keys.txt'
        self.statefile = self.prefix + '/state.gz'

        self.cofile = self.prefix + '/co-occur.txt'
        self.namefile = self.prefix + '/names.tsv'
        self.scorefile = self.prefix + '/scores.txt'

        self.mallet_corpus = self.prefix + '/corpus.mallet'

        if os.path.exists(self.tkfile):
            num_topics = len(open(self.tkfile).readlines())
            print('Read', num_topics, 'topics.')

        self.topics = [{} for i in range(num_topics)]
        self.params = [0 for i in range(num_topics)]
        self.word_count = 0

        if not os.path.exists(self.wtfile) or not os.path.exists(self.dtfile):
            self.read(corpus, bigrams)
            self.train(num_topics, iters)

        self.load_keys()
        self.load_wt()
        self.load_dt()

        self.load_names()
        self.load_scores()



    def get_n_docs(self):
        return len(self.topic_doc[0])

    def read(self, corpus, bigrams=False):
        stop = StopLexicon()

        #cmd = [self.path, 'import-dir',
        #       '--input', corpus,
        #       '--output', self.mallet_corpus,
        #       '--remove-stopwords',
        #       '--extra-stopwords', stop.file,
        #       '--token-regex', '[^\\s]+']

        # Much better to use the tokenization process within MALLET
        # This is immediately apparent.
        if os.path.isdir(corpus) :
            import_cmd = 'import-dir'
        else :
            import_cmd = 'import-file'
        cmd = [self.path, import_cmd,
               '--input', corpus,
               '--output', self.mallet_corpus,
               '--remove-stopwords',
               '--extra-stopwords', stop.file]

        if bigrams:
            cmd += ['--keep-sequence-bigrams', '--gram-sizes 2']
        else:
            cmd += ['--keep-sequence']

        if subprocess.call(cmd) != 0:
            sys.stderr.write('Mallet ' + import_cmd + ' failed.\n')
            print(cmd, file=sys.stderr)
            sys.exit(1)


    def train(self, num_topics, iters):

        cmd = [self.path, 'train-topics',
               '--input', self.prefix + '/corpus.mallet',
               '--num-topics', str(num_topics),
               '--num-iterations', str(iters),
               '--num-threads', str(PROCESSES),
               '--output-doc-topics', self.dtfile,
               '--word-topic-counts-file', self.wtfile,
               '--output-model', self.omfile,
               '--inferencer-filename', self.inffile,
               '--output-topic-keys', self.tkfile,
               '--output-state', self.statefile]

        #cmd = [self.path, 'train-topics',
        #       '--input', self.prefix + 'corpus.mallet',
        #       '--num-topics', str(num_topics),
        #       '--num-iterations', str(iters),
        #       '--optimize-interval', str(OPTIMIZE_INTERVAL),
        #       '--num-threads', str(PROCESSES),
        #       '--output-doc-topics', self.dtfile,
        #       '--word-topic-counts-file', self.wtfile,
        #       '--output-model', self.omfile,
        #       '--inferencer-filename', self.inffile,
        #       '--output-topic-keys', self.tkfile,
        #       '--output-state', self.statefile,
        #       '--beta', '0.00386']

        if subprocess.call(cmd) != 0:
            sys.stderr.write('Mallet train-topics failed.\n')
            sys.exit(1)


    def infer_topics(self, corpus, iters=1000):
        # Read corpus using the original corpus file as a pipe to ensure
        # compatability.
        cmd = [self.path, '/import-dir',
               '--input', corpus,
               '--output', self.mallet_corpus + '-infer',
               '--use-pipe-from', self.mallet_corpus]
        if subprocess.call(cmd) != 0:
            sys.stderr.write('Mallet import-dir failed.\n')
            sys.exit(1)

        # Don't overwrite original.
        self.dtfile += '-infer'

        cmd = [self.path, 'infer-topics',
               '--inferencer', self.inffile,
               '--input', self.mallet_corpus + '-infer',
               '--output-doc-topics', self.dtfile,
               '--num-iterations', str(iters)]
        if subprocess.call(cmd) != 0:
            sys.stderr.write('Mallet infer-topics failed.\n')
            sys.exit(1)

        self.load_dt()


    def load_keys(self):
        """Read the Dirichlet parameters from the topic key file."""
        print('Loading key file.')
        for line in open(self.tkfile):
            fields = line.split()
            topic_num = int(fields[0])
            parameter = float(fields[1])
            self.params[topic_num] = parameter


    def load_wt(self):
        print('Loading word-topic file.')
        for line in open(self.wtfile):
            tokens = line.strip().split()
            word = tokens[1]
            for c in tokens[2:]:
                topic, count = c.split(':')
                self.topics[int(topic)][word] = int(count)

        with open(self.wtkfile, 'w') as out:
            for topic in range(len(self.topics)):
                out.write('\t'.join([str(topic)] +
                                    [str(y) + '\t' + str(z) for (y, z) in
                                     self.topic_pairs(topic)[:20]]) + '\n')


    def load_dt(self):
        print('Loading document-topic composition file.')


        num_topics = len(self.topics)
        self.topic_doc = [[] for i in range(num_topics)]
        self.co_occur = zeros((num_topics, num_topics), int)

        # We need a cut-off for a topic to count as non-trivially occurring
        # in a document, and this needs to vary depending on the number of
        # topics. Based on experiments with 20 and 200 topic models, I chose
        # the thresholds (20, 0.3) and (200, 0.1) and fit the line
        #    y = -1/900*x + 290/900
        # with a min of 0.01. This is a preliminary measure and should be
        # adjusted for other corpora.
        thresh = max((290.0 - num_topics)/900.0, 0.01)

        for line in open(self.dtfile):
            row = line.strip().split()
            if row[0][0] == '#':
                continue
            if len(row) < 2:
                print('Error with composition row', row, file=sys.stderr)
                continue
            try:
                base = re.search(r'([^/]+)\.(xml|txt)$', row[1]).group(1)
            except:
                continue


            # Mallet's new format: The weight for each topic,
            # ordered by topic ID.
            topics = [(a, float(b)) for (a, b) in
                enumerate(row[2:])]

            # Read into document topic breakdown information.
            for topic_id, percent in topics:
                self.topic_doc[topic_id].append((base, percent))

            # Read into co-occurrence matrix.
            filt_topics = [(a, b) for (a, b) in topics if b > thresh]
            for topic_pair in combinations(filt_topics, 2):
                i1 = topic_pair[0][0]
                i2 = topic_pair[1][0]
                # Symmetric matrix.
                self.co_occur[i1][i2] += 1
                self.co_occur[i2][i1] += 1

        with open(self.cofile, 'w') as out:
            for row in self.co_occur:
                for c in row:
                    out.write('%s ' % (c))
                out.write('\n')

    def load_dt_into_es(self, embedding_file):

        # es.indices.delete(index='scidt', ignore=[400, 404])

        index_exists = self.es.indices.exists(index=["scidt"], ignore=404)

        if (index_exists is False):

            rep_min = 10000
            rep_max = -10000
            shape = 0

            i = 0
            count = 0
            length = 0
            start = time.time()

            for x in gzip.open(args.repfile):

                x_parts = x.strip().split()

                if (len(x_parts) == 2):
                    count = x_parts[0]
                    shape = x_parts[1]
                    continue

                minimum = min(float(xx) for xx in x_parts[1:])
                if (minimum < rep_min):
                    rep_min = minimum
                maximum = max(float(xx) for xx in x_parts[1:])
                if (maximum > rep_max):
                    rep_max = maximum
                i = i + 1
                if (i % 100000 == 0):
                    print
                    "it: " + str(i) + ", t=" + str(time.time() - start) + " s"

            self.es.indices.create(index='scidt', ignore=400)

            # Mapping to make the encoding of individual words unique.
            mapping_body = {
                "properties": {
                    "word": {
                        "type": "string",
                        "index": "not_analyzed"
                    }
                }
            }
            self.es.indices.put_mapping("rep", mapping_body, "scidt")

            # NOTE the (...) round brackets. This is for a generator.
            gen = ({
                "_index": "scidt",
                "_type": "rep",
                "_id": i,
                "_source": es_d,
            } for i, es_d in self.decode_ref_file(embedding_file))
            helpers.bulk(self.es, gen)

            actions = [{
                "_index": "scidt",
                "_type": "meta",
                "_id": 0,
                "_source": {
                    "rep_min": str(rep_min),
                    "rep_max": str(rep_max),
                    "rep_shape": str(shape)
                }
            }]
            print
            actions
            helpers.bulk(self.es, actions)

        meta = self.es.search(index="scidt", doc_type=['meta'],
                              body={"query": {
                                  "match_all": {}
                              }})

        # Note that if we've just built the index, it doesn't immediately provide a response
        # So we search and wait until it provides data.
        while (len(meta['hits']['hits']) == 0):
            time.sleep(5)
            meta = self.es.search(index="scidt", doc_type=['meta'],
                                  body={"query": {
                                      "match_all": {}
                                  }})

        meta_dict = meta['hits']['hits'][0]['_source']
        self.rep_min = float(meta_dict['rep_min'])
        self.rep_max = float(meta_dict['rep_max'])
        self.rep_shape = int(meta_dict['rep_shape']),
        self.numpy_rng = numpy.random.RandomState(12345)

    def load_names(self):
        """Load topic names from disk, if they exist. Otherwise, set
        every topic's name to its first three elements."""

        num_topics = len(self.topics)

        if not os.path.exists(self.namefile):

            # CHANGED BY GULLY
            self.names = [' '.join(n[0] for n in sorted(self.topics[i].items(),
                                                        key=operator.itemgetter(1),
                                                        reverse=True)[:3])
                          for i in range(num_topics)]

            return

        print('Loading topic names.')

        self.names = [''] * num_topics
        for line in open(self.namefile):
            topic, name = line.strip().split('\t', 1)
            if topic == 'Topic':
                continue
            self.names[int(topic)] = name


    def load_scores(self):
        """Load the topic scores from disk, if they exist. Otherwise,
        set every topic score to 1.0."""

        if not os.path.exists(self.scorefile):
            self.scores = [1.0 for x in self.topics]
            return

        print('Loading topic scores.')

        self.scores = []
        for line in open(self.scorefile):
            if line.startswith('Average'):
                continue
            self.scores.append(float(line))


    def topic_pairs(self, topic):
        return sorted(self.topics[topic].items(),
                      key=lambda x: (-1.0 * x[1], x[0]))

    def get_top_topic_words(self, t_id, n_words):
        topic_unigrams = defaultdict(set)

        sorted()

        for word in self.topics[t_id].keys():
            count = self.topics[t_id].get(word)
            word = tokens[1]
            counts = tokens[2:]
            for c in counts:
                (topic, count) = c.split(':')
                topic_unigrams[int(topic)].add((word, float(count)))

        unigrams = []
        for topic in topic_unigrams.keys():
            unigrams.append( sorted(topic_unigrams[topic], key=lambda x: x[1],reverse=True)[:100])