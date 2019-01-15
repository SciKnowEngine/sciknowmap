import sys
import os.path
import argparse
import urllib
import json

# Following https://developers.google.com/books/docs/v1/getting_started?csw=1
# We run queries over all terms in the basic taxonomy against the Google Books Volume collection
# and convert them to the Erudite Learning Resource schema

def add_book_lists(book_ids, term_data, new_term_data):
    for item in new_term_data:
        if (item['id'] not in book_ids):
            term_data.append(item)
            book_ids.append(item['id'])
    return (book_ids, term_data)

def run_search(term, addl_query):
    query = BASE_URL + '?q=' + term + '&maxResults=40&key=' + args.apiKey \
            + addl_query

    print('\n' + term + '\n' + query)

    google_books_response = urllib.urlopen(query)

    google_books_response_string = google_books_response.read()
    google_books_data = json.loads(google_books_response_string)
    term_data = google_books_data['items']

    total = google_books_data['totalItems']

    for i in range(40, total, 40):

        query = BASE_URL + '?q=' + term + '&maxResults=40&key=' + args.apiKey + '&startIndex=' + str(i) \
                + addl_query
        google_books_response = urllib.urlopen(query)

        google_books_response_string = google_books_response.read()
        google_books_data = json.loads(google_books_response_string)

        if (google_books_data.get('items', None) is not None):
            for item in google_books_data['items']:
                term_data.append(item)
        else:
            return term_data

        sys.stdout.write('\r')
        sys.stdout.write("[%-20s] %d%% %d/%d" % ('='*(i/total*20), (100 * i / total), i, total))
        sys.stdout.flush()
        sys.stdout.write("")

    return term_data

def nmxl_files(members):
    try:
        for tarinfo in members:
            if os.path.splitext(tarinfo.name)[1] == ".nxml":
                yield tarinfo
    except IOError:
        doNothing = None

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inFile', help='Input file of search terms (one per line)')
    parser.add_argument('-o', '--outDir', help='Output directory')
    parser.add_argument('-k', '--apiKey', help='API Key')

    args = parser.parse_args()

    # We construct searches based on the BASE_URL + args.apiKey + query
    # https://www.googleapis.com/apiName/apiVersion/resourcePath?parameters
    BASE_URL = 'https://www.googleapis.com/books/v1/volumes'

    if(args.inFile is None):
        sys.exit("USAGE: python 00_call_google_books_api.py -i terms_file -o out_dir")

    if( args.outDir[-1:] != '/' ):
        args.outDir += '/'

    with open(args.inFile) as f:
        term_list = f.readlines()

    if( os.path.exists(args.outDir) is False ):
        os.mkdir(args.outDir)

    #with open('google_query_count.dat') as f:
    #    query_count = f.readline()

    term_data = []
    book_ids = []

    for term in term_list:

        term = term.strip()

        if(os.path.isfile(args.outDir + term + '.json')):
                continue

        try:

            term_data = []
            book_ids = []

            term_data_1 = run_search(term, '&printType=books&orderBy=newest')
            (book_ids, term_data) = add_book_lists(book_ids, term_data, term_data_1)

            term_data_2 = run_search(term, '&printType=books&orderBy=relevance')
            (book_ids, term_data) = add_book_lists(book_ids, term_data, term_data_2)

            term_data_3 = run_search(term, '&printType=books&filter=free-ebooks&orderBy=newest')
            (book_ids, term_data) = add_book_lists(book_ids, term_data, term_data_3)

            term_data_4 = run_search(term, '&printType=books&filter=free-ebooks&orderBy=relevance')
            (book_ids, term_data) = add_book_lists(book_ids, term_data, term_data_4)

            out_data = {'kind':'books#volumes', 'totalItems':len(term_data)}
            out_data['items'] = term_data
            with open(args.outDir + term + '.json', 'w') as output:
                out_string = json.dumps(out_data, indent=4, separators=(',', ': '))
                output.write(out_string)

        except Exception:

            print(term + ' failed')
            print(sys.exc_info()[0])
            continue


