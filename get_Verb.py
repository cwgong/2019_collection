import numpy as np

open_words1 = open('C:\\Users\\Administrator\\Desktop\\第一个动词.txt','r',encoding='utf-8')
feat_words1 = open_words1.readlines()
# feat_word1 = feat_words1.split(" ")

open_words2 = open('C:\\Users\\Administrator\\Desktop\\第二个动词.txt','r',encoding='utf-8')
feat_words2 = open_words2.readlines()
# feat_word2 = feat_words2.split(" ")

open_words3 = open('C:\\Users\\Administrator\\Desktop\\第三个动词.txt','r',encoding='utf-8')
feat_words3 = open_words3.readlines()
# feat_word3 = feat_words3.split(" ")

all_s1 = []
all_s2 = []
all_s3 = []

all1_s1 = []
all2_s2 = []
all3_s3 = []

# print(feat_words1)

for p in feat_words1:
    p.repalce(")",'')
    p.repalce("(",'')
    p.repalce("'",'')
    p.repalce(" ",'')
    p.repalce(",",'')
    all_s1.append(p.split()[0])
    all1_s1.append(p.split()[1])

for p in feat_words2:
    all_s2.append(p.split()[0])
    all2_s2.append(p.split()[1])

for p in feat_words3:
    all_s3.append(p.split()[0])
    all3_s3.append(p.split()[1])

for i in all1_s1:
    print(i)
