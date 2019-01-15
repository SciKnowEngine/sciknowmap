#!/usr/bin/env python3

# TechKnAcq: Concept Graph
# Jonathan Gordon

import codecs
import csv
import math
import os
import re
import sys
import pandas as pd
from collections import defaultdict
from itertools import combinations

import click
from numpy import zeros


@click.command()
@click.argument('in_file', type=click.Path(exists=True))
@click.argument('out_file', type=click.STRING)
def main(in_file, out_file):

    tsv = pd.read_csv(in_file, sep='\t')
    c_s_lookup = {}
    c_p_lookup = {}
    s_c_lookup = {}
    p_c_lookup = {}

    clause_max = -1
    clause_min = 1000

    for i, row in tsv.iterrows():
        t = row['topic']
        vb = row['column']
        vl = row['value']

    output = open(out_file, 'w')
    output.write(html)
    output.close()

if __name__ == '__main__':
    main()
