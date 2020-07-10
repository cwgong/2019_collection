# -*- coding: utf-8 -*-

import os
import io
import json
import requests
import time_utils
import codecs


def get_cluster_info_from_api(start_time, end_time):
    clusterIds = []
    url = "http://information-doc-service:31001/cluster/search"

    totalCount = 0

    try:
        cp = 1
        while True:
            params = dict()
            params["cp"] = cp
            params["ps"] = 50
            params["clusterTypes"] = ['热点事件']
            params["delFlag"] = 0
            params["timeField"] = 'publishAt'
            params["startAt"] = start_time
            params["endAt"] = end_time

            r = requests.get(url, params)
            result_json = r.json()

            # print(result_json)
            # break
            totalCount = result_json['data']['totalCount']

            if len(result_json['data']['list']) == 0:
                break
            for item in result_json['data']['list']:
                # title
                # hot
                cluster_id = item['id']
                keywords = item['keywords']
                createAt = item['createAt']
                publishAt = item['publishAt']
                clusterIds.append(cluster_id)
            cp += 1
    except Exception as e:
        # logger.exception("Exception: {}".format(e))
        print("Exception: {}".format(e))

    # logger.info("get_cluster_info_from_api count: {}".format(totalCount))
    print("get_cluster_info_from_api count: {}".format(totalCount))
    return clusterIds

def write_to_file(string, file_path, param):
    out = codecs.open(file_path, param, 'utf-8')
    out.write(string)
    out.close


def get_cluster_infoids_feature_from_api(clusterIds):
    url = 'http://index-information-service:31001/information/relation/search'

    totalCount = 0

    # 由于 clusterIds 太多，分批取
    batch_size = 50
    epoch = int(len(clusterIds) / batch_size) + 1
    begin = 0

    # 遍历取数据
    for i in range(0, epoch):

        clusterIds_ = clusterIds[begin: begin + batch_size]
        begin += batch_size

        try:
            cp = 1
            while True:
                params = dict()
                params["cp"] = cp
                params["ps"] = 50
                params["clusterIds"] = clusterIds_
                params["delFlag"] = 0
                params["relationTypes"] = "事件概述,事件影响"
                params["human"] = 0
                params["startAt"] = start_time
                params["endAt"] = end_time

                r = requests.get(url, params)
                result_json = r.json()

                # print(result_json)
                # break
                batch_totalCount = result_json['data']['totalCount']
                totalCount += batch_totalCount
                if len(result_json['data']['list']) == 0:
                    break
                for item in result_json['data']['list']:
                    # contentId
                    # id
                    # editAt
                    # mediaFrom
                    # 'relationType': ['事件影响']
                    # title
                    cluster_id = item['clusterId']
                    content = item['content']
                    createAt = item['createAt']
                    publishAt = item['publishAt']
                    relationType = item['relationType']
                    if '事件影响' in relationType:
                        print(content)
#                        write_to_file(content, 'C:\\Users\\Administrator\\Desktop\\no_1.txt', 'r+')
                        out = codecs.open('C:\\Users\\Administrator\\Desktop\\初始文本.txt', 'r+', 'utf-8')
                        out.read()
                        out.write('\n')
                        out.write('\n')
                        out.write('\n')
                        out.write(content)
                        out.close
                        print("")
                cp += 1
        except Exception as e:
            # logger.exception("Exception: {}".format(e))
            print("Exception: {}".format(e))

    # logger.info("get_cluster_infoids_feature_from_api count: {}".format(totalCount))
    print("get_cluster_infoids_feature_from_api count: {}".format(totalCount))


if __name__ == '__main__':
    start_time = time_utils.n_days_ago_milli_time(100)
    end_time = time_utils.current_milli_time()
    clusterIds = get_cluster_info_from_api(start_time, end_time)
    get_cluster_infoids_feature_from_api(clusterIds)
