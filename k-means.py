import os
import io
import json
import requests
import time_utils


# def get_cluster_info_from_api(start_time, end_time):
#     clusterIds = []
#     url = "http://information-doc-service:31001/cluster/search"
#
#     totalCount = 0
#
#     try:
#         cp = 1
#         while True:
#             params = dict()
#             params["cp"] = cp
#             params["ps"] = 50
#             params["clusterTypes"] = ['热点事件']
#             params["delFlag"] = 0
#             params["timeField"] = 'publishAt'
#             params["startAt"] = start_time
#             params["endAt"] = end_time
#
#             r = requests.get(url, params)
#             result_json = r.json()
#
#             # print(result_json)
#             # break
#             totalCount = result_json['data']['totalCount']
#
#             if len(result_json['data']['list']) == 0:
#                 break
#             for item in result_json['data']['list']:
#                 # title
#                 # hot
#                 cluster_id = item['id']
#                 keywords = item['keywords']
#                 createAt = item['createAt']
#                 publishAt = item['publishAt']
#                 clusterIds.append(cluster_id)
#             cp += 1
#     except Exception as e:
#         # logger.exception("Exception: {}".format(e))
#         print("Exception: {}".format(e))
#
#     # logger.info("get_cluster_info_from_api count: {}".format(totalCount))
#     print("get_cluster_info_from_api count: {}".format(totalCount))
#     return clusterIds
#
#
# def get_cluster_infoids_feature_from_api(clusterIds):
#     url = 'http://index-information-service:31001/information/relation/search'
#
#     totalCount = 0
#
#     # 由于 clusterIds 太多，分批取
#     batch_size = 50
#     epoch = int(len(clusterIds) / batch_size) + 1
#     begin = 0
#
#     # 遍历取数据
#     for i in range(0, epoch):
#
#         clusterIds_ = clusterIds[begin: begin + batch_size]
#         begin += batch_size
#
#         try:
#             cp = 1
#             while True:
#                 params = dict()
#                 params["cp"] = cp
#                 params["ps"] = 50
#                 params["clusterIds"] = clusterIds_
#                 params["delFlag"] = 0
#                 params["relationTypes"] = "事件概述,事件影响"
#                 params["human"] = 0
#                 params["startAt"] = start_time
#                 params["endAt"] = end_time
#
#                 r = requests.get(url, params)
#                 result_json = r.json()
#
#                 # print(result_json)
#                 # break
#                 batch_totalCount = result_json['data']['totalCount']
#                 totalCount += batch_totalCount
#                 if len(result_json['data']['list']) == 0:
#                     break
#                 for item in result_json['data']['list']:
#                     # contentId
#                     # id
#                     # editAt
#                     # mediaFrom
#                     # 'relationType': ['事件影响']
#                     # title
#                     cluster_id = item['clusterId']
#                     content = item['content']
#                     createAt = item['createAt']
#                     publishAt = item['publishAt']
#                     relationType = item['relationType']
#                     if '事件影响' in relationType:
#                         print(content)
#                         print(relationType)
#                 cp += 1
#         except Exception as e:
#             # logger.exception("Exception: {}".format(e))
#             print("Exception: {}".format(e))
#
#     # logger.info("get_cluster_infoids_feature_from_api count: {}".format(totalCount))
#     print("get_cluster_infoids_feature_from_api count: {}".format(totalCount))


def get_cluster_info_from_features_api(start_time, end_time):
    clusterIds = []
    infoid = []
    url = "http://index-information-service:31001/cluster/information/search?clusterIds=d1c500592d8a22ac273c2f4662d0c21f"

    totalCount = 0

    try:
        cp = 1
        while True:
            params = dict()
            params["cp"] = cp
            params["ps"] = 50
            params["clusterTypes"] = ['热点事件']
            params["delFlag"] = 0
            params["timeField"] = "publishAt"
            params["startAt"] = start_time
            params["endAt"] = end_time

            r = requests.get(url, params)
            result_json = r.json()
            print(result_json)
            # break
            totalCount = result_json['data']['totalCount']

            if len(result_json['data']['list']) == 0:
                break
            for item in result_json['data']['list']:
                # title
                # hot
                infoid1 = item['infoid']
                # keywords = item['keywords']
                createAt = item['createAt']
                publishAt = item['publishAt']
                infoid.append(infoid1)
                print(infoid)
            cp += 1
    except Exception as e:
        # logger.exception("Exception: {}".format(e))
        print("Exception: {}".format(e))
        print("chucuo")

    # logger.info("get_cluster_info_from_api count: {}".format(totalCount))
    print("get_cluster_info_from_features_api count: {}".format(totalCount))
    return infoid

def get_cluster_infoid_feature_from_api(clusterIds):
    # url = "http://information-doc-service:31001/information/detail?ids=f2400ef6a2c0dba0aa82779a49402ba4"
    #
    # totalCount = 0
    #
    # # 由于 clusterIds 太多，分批取
    # batch_size = 50
    # epoch = int(len(clusterIds) / batch_size) + 1
    # begin = 0
    #
    # # 遍历取数据
    # for i in range(0, epoch):
    #
    #     clusterIds_ = clusterIds[begin: begin + batch_size]
    #     begin += batch_size
    #
    #     try:
    #         cp = 1
    #         while True:
    #             params = dict()
    #             params["cp"] = cp
    #             params["ps"] = 50
    #             params["clusterIds"] = clusterIds_
    #             params["delFlag"] = 0
    #             params["relationTypes"] = "事件概述,事件影响"
    #             params["human"] = 0
    #             params["startAt"] = start_time
    #             params["endAt"] = end_time
    #
    #             r = requests.get(url, params)
    #             result_json = r.json()
    #
    #             # print(result_json)
    #             # break
    #             batch_totalCount = result_json['data']['totalCount']
    #             totalCount += batch_totalCount
    #             if len(result_json['data']['list']) == 0:
    #                 break
    #             for item in result_json['data']['list']:
    #                 # contentId
    #                 # id
    #                 # editAt
    #                 # mediaFrom
    #                 # 'relationType': ['事件影响']
    #                 # title
    #                 cluster_id = item['clusterId']
    #                 content = item['content']
    #                 createAt = item['createAt']
    #                 publishAt = item['publishAt']
    #                 relationType = item['relationType']
    #                 if '事件影响' in relationType:
    #                     print(content)
    #                     print(relationType)
    #             cp += 1
    #     except Exception as e:
    #         # logger.exception("Exception: {}".format(e))
    #         print("Exception: {}".format(e))
    #
    # # logger.info("get_cluster_infoids_feature_from_api count: {}".format(totalCount))
    # print("get_cluster_infoids_feature_from_api count: {}".format(totalCount))

    url = "http://information-doc-service:31001/information/detail?ids=ab81dc6e7c5b9f179f08c1ce414514d2"
    for item in clusterIds:
        params = dict()
        params['ids'] = item

        r = requests.get(url,params)
        result_json = r.json()


    print(result_json)

if __name__ == '__main__':
    start_time = time_utils.n_days_ago_milli_time(1)
    end_time = time_utils.current_milli_time()
    # clusterIds = get_cluster_info_from_api(start_time, end_time)
    # get_cluster_infoids_feature_from_api(clusterIds)
    clusterIds = get_cluster_info_from_features_api(start_time, end_time)
    # get_cluster_infoid_feature_from_api(clusterIds)