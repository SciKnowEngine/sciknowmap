import re
import sys
import codecs
import os.path
import argparse
import urllib
import json

#sys.path.append('../../..')
print("\n".join(sys.path))

# Following https://developers.google.com/books/docs/v1/getting_started?csw=1
# We run queries over all terms in the basic taxonomy against the Google Books Volume collection
# and convert them to the Erudite Learning Resource schema

def read_resources_from_dir(inDir):

    resources = {}
    for root, dirs, files in os.walk(inDir):
        for file in files:
            json_lines = []
            if os.path.isfile(root + '/' + file) and file[-5:] == '.json':

                with open(root + '/' + file, 'r') as json_file:
                    json_lines = json_file.readlines()

                json_str = ""
                for l in json_lines:
                    json_str += l
                json_data = json.loads(json_str)

                count = 0
                for book_item in json_data['items']:

                    course = {}
                    course['id'] = book_item['id']
                    if( book_item['volumeInfo'].get('title', None) is None):
                        continue
                    if (book_item['volumeInfo'].get('description', None) is None):
                        continue
                    course['title'] = book_item['volumeInfo']['title']
                    course['description'] = book_item['volumeInfo']['description']
                    course['url'] = book_item['volumeInfo']['infoLink']
                    if (book_item.get('searchInfo', None) is not None):
                        course['slug'] = book_item['searchInfo']['textSnippet']
                    else:
                        course['slug'] = ""
                    if (book_item.get('saleInfo', None) is not None):
                        if (book_item['saleInfo'].get('listPrice', None) is not None):
                            course['price'] = book_item['saleInfo']['listPrice']
                    course['language'] = book_item['volumeInfo']['language']
                    course['format'] = 'video',
                    #course.license = content['license']
                    #course.venue = book_item['volumeInfo']['publisher']
                    channel = {}
                    if (book_item.get('searchInfo', None) is not None):
                        course['slug'] = book_item['searchInfo']['textSnippet']
                    if( book_item['volumeInfo'].get('publisher', None) is not None):
                        channel['id'] = book_item['volumeInfo']['publisher']
                        channel['name'] = book_item['volumeInfo']['publisher']
                    if( book_item['volumeInfo'].get('photo', None) is not None):
                        course['photo'] = book_item['volumeInfo']['imageLinks']['thumbnail']
                    #if content['uploader_id'] is not None:
                    #    channel.url = 'https://www.youtube.com/channel/' + content['uploader_id']
                    course['provider'] = [channel]
                    tag = {}
                    tag['concept_tag'] = file
                    course['tags'] = [tag]

                    resources[book_item['id']] = course
                    count += 1

                print('{0:s}\t{1:d}'.format(file, count))

    return resources