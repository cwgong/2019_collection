import numpy
import types
import matplotlib.pyplot as plt

def file2matrix(filename):
    fr = open(filename,encoding='utf-8')
    arrayOLines = fr.readline()
    numberOfLines = len(arrayOLines)
    content = arrayOLines[1]
    print(content)

def dict_get(dict,objkey,default):
    tmp = dict
    for k in tmp.key():
        if k == objkey:
            return k

