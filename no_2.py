# -*- coding: utf-8 -*-

import os
import io
import json
import requests


def split_sentence(sen):
    nlp_url = 'http://hanlp-rough-service:31001/hanlp/segment/rough'
    try:
        cut_sen = dict()
        cut_sen['content'] = sen
        data = json.dumps(cut_sen).encode("UTF-8")
        cut_response = requests.post(nlp_url, data=data)
        cut_response_json = cut_response.json()
        return cut_response_json['data']
    except Exception as e:
        # logger.exception("Exception: {}".format(e))
        print(e)
        return []


def fetch_data(text_file):
    raw_data = []
    if not os.path.exists(text_file):
        return raw_data

    count = 0
    with io.open(text_file, "r", encoding='utf-8') as f:
        while True:
            line = f.readline()
            if len(line) > 0:
                try:
                    json_data = json.loads(line)
                    if 'title' in json_data.keys() and 'content' in json_data:
                        count += 1
                        title = json_data['title']
                        title_seg = split_sentence(title)

                        print(title)
                        print(title_seg)
                        raw_data.append(title)
                        break
                except Exception as e:
                    continue
            else:
                break
    return raw_data


fetch_data('C:\\Drivers\\123\\2018-10-01.txt')
