import json
import requests
import os
import numpy as np

open_words = open('C:\\Users\\Administrator\\Desktop\\关键词.txt','r',encoding='utf-8')
feat_words = open_words.read()
#print(feat_words)
feat_word = feat_words.split(" ")



def split_sentence(sen):
    nlp_url = 'http://hanlp-rough-service:31001//hanlp//segment//rough'
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

paragraphs = []
f = open('C:\\Users\\Administrator\\Desktop\\初始文本.txt', encoding='utf-8')
paragraphs = f.readlines()

f.close()
useful_s1 = []
all_s = []
securities = []
for p in paragraphs:
    a = p.split("。")
    for b in a:
        if len(b.strip()) != 0:
            all_s.append(b.strip())                           #将所有的句子安句号分开之后加入到all_s列表里面

useful_s = []
for s in all_s:                                                 #从列表里面取出来一句
    c1 = 0
    for security in securities:
        if security in s:
            c1 += 1

    c2 = 0
    data = split_sentence(s)                                    #返回的是字典类型，返回的是那一句的分词结果
    natures = [x['nature'] for x in data]
    for nature in natures:                                      #取出分析师那一类的特征词
        if nature == 'fxs':
            c2 += 1

    if c1 !=0 or c2 !=0:
        useful_s.append(s)
# print(useful_s)
# for s in useful_s:
#    data = split_sentence(s)
#    words = [x['word'] for x in data]
#    natures = [x['nature'] for x in data]
#
#    string = ''
#
#    for i in range(0,len(words)):
#       string += words[i]+'/'+natures[i]+' '
#    print(string)

#for i in range(0,len(feat_word)):
#    print(feat_word[i])

# c4 = 0
# for s1 in useful_s:
#     c3 = 0
#     data=split_sentence(s1)
#     words = [x['word'] for x in data]
#     natures = [x['nature'] for x in data]
#
#     for word in words:
#         if word in feat_word:
#             c3 = c3 + 1
#
#     if(c3 > 0):
#         useful_s1.append(s1)

# for s3 in useful_s1:
#     print(s3)
# j=0
# user = []
# for s4 in useful_s1:
#     data = split_sentence(s4)
#     words = [x['word'] for x in data]
#     natures = [x['nature'] for x in data]
#
#     string = ''
#     # for i in range(0,len(words)):
#     #     string += words[i]+'/'+natures[i]+' '
#     # print(string)
#
#     for i in range(0,len(words)):
#         if words[i] in feat_word:
#             j=i
#             break
#     while(True):
#         if natures[j] == 'fxs':
#             break
#         if natures[j] == 'nr':
#             break
#         j = j-1
#
#     user.append(words[j])
#
# user = list(set(user))

# for i in range(0,len(user)):
#     print(user[i])

verb1 = []
verb2 = []
verb3 = []

for i in range(0,len(useful_s)):
    data = split_sentence(useful_s[i])
    words = [x['word'] for x in data]
    natures = [x['nature'] for x in data]
#
#     for j in range(0,len(useful_s[i])):
#         if natures[j] == 'fxs':
#             while(True):
#                 if(j >= len(natures)):
#                     break
#                 if(natures[j] == 'v'):
#                     verb1.append(words[j])
#                     j = j+1
#                     while(True):
#                         if(j >= len(natures)):
#                             break
#                         if(natures[j] == 'v'):
#                             verb2.append(words[j])
#                             j = j+1
#                             while(True):
#                                 if (j >= len(natures)):
#                                     break
#                                 if (natures[j] == 'v'):
#                                     verb3.append(words[j])
#                                     break
#                                 else:
#                                     j = j+1
#                             break
#                         else:
#                             j = j+1
# #                    print(words[j])
#                     break
#                 else:
#                     j = j+1
#             break

    for j in range(0,len(useful_s[i])):
        if natures[j] == 'fxs':
            j = j+1
            while(True):
                if(j >= len(natures)):
                    break
                else:
                    verb1.append(words[j])
                    j = j+1
                    if (j >= len(natures)):
                        break
                    else:
                        verb2.append(words[j])
                        j = j+1
                        if (j >= len(natures)):
                            break
                        else:
                            verb3.append(words[j])
                            break

            break

# for i in range(0,len(verb1)):
#     print(verb1[i])

verb1_set1 = list(set(verb1))                         #用来输出fxs后面几个动词的列表
verb1_sets1 = [0]*len(verb1_set1)

for word in verb1:
    if word in verb1_set1:
        verb1_sets1[verb1_set1.index(word)] += 1

verb1_dict = dict(map(lambda x,y:[x,y], verb1_set1,verb1_sets1))
source_count_sort1 = sorted(verb1_dict.items(), key=lambda d: d[1], reverse=True)

verb1_set2 = list(set(verb2))
verb1_sets2 = [0]*len(verb1_set2)

for word in verb2:
    if word in verb1_set2:
        verb1_sets2[verb1_set2.index(word)] += 1

verb2_dict = dict(map(lambda x,y:[x,y], verb1_set2,verb1_sets2))
source_count_sort2 = sorted(verb2_dict.items(), key=lambda d: d[1], reverse=True)

verb1_set3 = list(set(verb3))
verb1_sets3 = [0]*len(verb1_set3)

for word in verb3:
    if word in verb1_set3:
        verb1_sets3[verb1_set3.index(word)] += 1

verb3_dict = dict(map(lambda x,y:[x,y], verb1_set3,verb1_sets3))
source_count_sort3 = sorted(verb3_dict.items(), key=lambda d: d[1], reverse=True)

print("fxs后面第一个词：")
for word in source_count_sort1:
    print(word)

print("fxs后面第二个词：")
for word in source_count_sort2:
    print(word)

print("fxs后面第三个词：")
for word in source_count_sort3:
    print(word)

# print("fxs后第一个动词：")
# for i in range(0,len(verb1_sets1)):
#     print(verb1_set1[i]+'    '+str(verb1_sets1[i]))
#
# print("fxs后第二个动词：")
# for i in range(0,len(verb1_sets2)):
#     print(verb1_set2[i]+'    '+str(verb1_sets2[i]))
#
# print("fxs后第三个动词：")
# for i in range(0,len(verb1_sets3)):
#     print(verb1_set3[i]+'    '+str(verb1_sets3[i]))



# use_verb = []
# for i in range(0,len(verb1_sets)):                                      #筛选出词频大于5的
#     if(int(verb1_sets[i]) > 5):
#         use_verb.append((verb1_set[i]+'   '+str(verb1_sets[i])))
#
# for i in range(0,len(use_verb)):
#         print(use_verb[i])

# words_1 = []
# words_2 = []
# words_3 = []
#
# for s_n in useful_s:
#     data = split_sentence(s_n)
#     words = [x['word'] for x in data]
#     natures = [x['nature'] for x in data]
#
#     for j in range(0,len(natures)):
#  #       print(j)
#         if natures[j] == 'fxs':
# #            print(j)
#             j = j-1
#             if(j >= 0):
#                 words_1.append(words[j])
#                 j = j-1
#                 if(j >= 0):
#                     words_2.append(words[j])
#                     j = j-1
#                     if(j >= 0):
#                         words_3.append(words[j])
#                     else:
#                         break
#                 else:
#                     break
#             else:
#                 break

# for i in range(0,len(words_1)):
#     print(words_1[i])
#
# for i in range(0,len(words_2)):
#     print(words_2[i])
#
# for i in range(0,len(words_3)):
#     print(words_3[i])

# word_1 = []
# word_2 = []
# word_3 = []
# word_1Num = []
# word_2Num = []
# word_3Num = []
#
# word_1 = list(set(words_1))
# word_2 = list(set(words_2))
# word_3 = list(set(words_3))
#
# word_1Num = [0]*len(word_1)
# word_2Num = [0]*len(word_2)
# word_3Num = [0]*len(word_3)
#
# for word in words_1:
#     if word in word_1:
#        word_1Num[word_1.index(word)] += 1
#
# for word in words_2:
#     if word in word_2:
#        word_2Num[word_2.index(word)] += 1
#
# for word in words_3:
#     if word in word_3:
#        word_3Num[word_3.index(word)] += 1
#
# print("fxs前面的第一个词:")
#
# for i in range(0,len(word_1)):
#     print(word_1[i]+'   '+str(word_1Num[i]))
#
# print("fxs前面的第二个词:")
#
# for i in range(0,len(word_2)):
#     print(word_2[i]+'   '+str(word_2Num[i]))
#
# print("fxs前面的第三个词:")
#
# for i in range(0,len(word_3)):
#     print(word_3[i]+'   '+str(word_3Num[i]))


