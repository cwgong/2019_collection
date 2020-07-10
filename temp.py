# import pdfkit
# import json
# import requests
# import re
# import io

# with io.open('kit.json','r',encoding='utf-8') as f:
#     dic = json.load(f)
# print(dic)

# dic = json.load('kit.json')

# with open('kit.json', encoding='utf-8') as f:
#     line = f.readline()
#     d = json.loads(line)
#     name = d['name']
#     company_url = d['company_url']
#     telephone = d['telphone']
#     crawl_time = d['crawl_time']
#     print(name, company_url, telephone, crawl_time)
#     f.close()

 #<class 'dict'>,JSON文件读入到内存以后，就是一个Python中的字典。
# with open('sharehold.json', 'r', encoding='utf-8') as f:
#     x = json.load(f)
#     print(x)

    # for item in f.readlines():
    #     item = re.sub('\'', '\"', item)
    #     dic = json.loads(item)
    #     print(dic)
    # att = []
    # while True:
    #
    #     line = f.readline()
    #     if line != '\n':
    #         line = line.strip("\n")
    #         att.append(line)
    # # d = json.loads(line)
    #     if len(line) == 0:
    #         break
#
# for item in att:
    # print(item)

# att = str(att)
# xx = att.rsplit('}')
# for ss in xx:
#     print(ss)

# dict = {
#     "content": "./IMAGE/annual/annual_094_0.jpg",
#     "footer": "down",
#     "header": "up",
#     "page_num": 94,
#     "type": "graph"
# }
# print(type(dict))
# print(dict['content'].split('/')[3])
# def is_title(text):
#     if '。' in text:
#         return -1
#     if text.endswith('；'):
#         return -1
#
#     rule_list = [r'^（[0-9]{1,2}）',  # （1）
#                  r'^[0-9]{1,2}）',  # 1）
#                  r'^[0-9]{1,2}、',  # 1、
#                  r'^[0-9]{1,2}\.',  # 1.
#                  r'^[0-9]{1,2}．',  # 1．
#                  r'^\([0-9]{1,2}\)',  # (1)
#                  r'^（[一二三四五六七八九十]+）',  # （一）
#                  r'^[一二三四五六七八九十]+、',  # 一、
#                  r'^\([一二三四五六七八九十]+\)',  # (一)
#                  r'^[①②③④⑤⑥⑦⑧⑨⑩⑪]+']  # ①
#
#     text = text.strip('\t').strip()
#     for i in range(len(rule_list)):
#         pattern = re.compile(rule_list[i])
#         result = pattern.findall(text)
#         if len(result) != 0:
#             return i
#     return -1
#
# s = '1.woshiasdnakmd'
# print(is_title(s))
# import os
#
# os.system("libreoffice --invisible --convert-to pdf --outdir ./ ./temp.docx")
# -*- coding: utf-8 -*-
"""
linux platform word to pdf
"""
import subprocess
import os

try:
    from comtypes import client
except ImportError:
    client = None

try:
    from win32com.client import constants, gencache
except ImportError:
    constants = None
    gencache = None


def doc2pdf_linux(docPath, pdfPath):
    """
    允许的文档格式：doc，docx
    仅在linux平台下可以
    需要在linux中下载好libreoffice
    """
    #  注意cmd中的libreoffice要和linux中安装的一致
    cmd = 'libreoffice6.2 --headless --convert-to pdf'.split() + [docPath] + ['--outdir'] + [pdfPath]
    # cmd = 'libreoffice6.2 --headless --convert-to pdf'.split() + [docPath]
    p = subprocess.Popen(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    p.wait(timeout=30)  # 停顿30秒等待转化
    stdout, stderr = p.communicate()
    if stderr:
        raise subprocess.SubprocessError(stderr)


def doc2pdf(docPath, pdfPath):
    """
    注意使用绝对路径
    pdf的生成只写路径，不写名字
    """
    docPathTrue = os.path.abspath(docPath)  # bugfix - searching files in windows/system32
    if client is None:#判断环境，linux环境这里肯定为None
        return doc2pdf_linux(docPathTrue, pdfPath)
    word = gencache.EnsureDispatch('Word.Application')
    doc = word.Documents.Open(docPathTrue, ReadOnly=1)
    doc.ExportAsFixedFormat(pdfPath,
                            constants.wdExportFormatPDF,
                            Item=constants.wdExportDocumentWithMarkup,
                            CreateBookmarks=constants.wdExportCreateHeadingBookmarks)
    word.Quit(constants.wdDoNotSaveChanges)


if __name__ == '__main__':
    wordpath='./temp.docx'
    pdfpath='./'
    doc2pdf(wordpath,pdfpath)

