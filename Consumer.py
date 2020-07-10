from kafka import KafkaConsumer
import os
import subprocess, sys
import psutil
import numpy as np
import cv2

from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.layout import LAParams
from pdfminer.converter import PDFPageAggregator
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.layout import LTTextBoxHorizontal
from pdfminer.pdfpage import PDFTextExtractionNotAllowed

from collections import Counter
from pdfminer.pdfinterp import resolve1
import operator
from difflib import SequenceMatcher
from pdfminer.layout import LAParams, LTImage, LTFigure, LTRect

import time
from threading import Thread
import multiprocessing

import pickle
import re


# ------------------------------------------------
def countsimilar(lists):
    new_list = []  # [txt,count]
    for i in lists:
        notmatch = True
        for idx, j in enumerate(new_list):
            if SequenceMatcher(None, i, j[0]).ratio() > 0.8:
                new_list[idx][1] += 1
                notmatch = False
        if notmatch:
            new_list.append([i, 1])
    return new_list


# ------------------------------------------------------------
def initial(out_dir):  # 初始化資料夾
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)


# -----------------------------------------------------------------------
class ThreadWithReturnValue(Thread):  # 能回傳結果的thread
    def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, *, daemon=None):
        Thread.__init__(self, group, target, name, args, kwargs, daemon=daemon)

        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)

    def join(self):
        Thread.join(self)
        return self._return


# -------------------------------------------------------------------
def make_pic(indir, filename, out_dir):  # 使用ghostscript 將pdf轉為每頁圖片
    name = filename.split('.')[0]
    initial(out_dir + '/' + name)
    pdf_route = '{}/{}.pdf'.format(indir, name)
    gs = 'gswin32c' if (sys.platform == 'win32') else 'gs'
    p = subprocess.Popen([gs,
                          '-dBATCH', '-dNOPAUSE', '-sDEVICE=jpeg', '-r144',
                          '-sOutputFile={}/{}/{}_%03d.jpg'.format(out_dir, name, name), pdf_route])
    p.wait()
    print('Extracting Image Done!')


# ------------------------------------------------------------------
def preprocess(src):  # 影像之前處理
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    outs = cv2.adaptiveThreshold(255 - gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(outs, cv2.MORPH_CLOSE, kernel)
    return closing


# ----------------------------------------------
def findbox(rct, maps):  # 尋找mask之中的物件是否為方形 利用確認上下左右是否都找得到值（若梯形就找不到
    isrect = True
    array = [(rct[0], rct[1]), (rct[0] + rct[2], rct[1]), (rct[0] + rct[2], rct[1] + rct[3]), (rct[0], rct[1] + rct[3])]
    for pt in array:
        fill = False
        jmin = pt[0] - 10 if pt[0] > 10 else 0
        jmax = pt[0] + 10 if pt[0] + 10 < maps.shape[1] else maps.shape[1]
        imin = pt[1] - 10 if pt[1] > 10 else 0
        imax = pt[1] + 10 if pt[1] + 10 < maps.shape[0] else maps.shape[0]

        for j in range(jmin, jmax):
            for i in range(imin, imax):
                if maps[i, j]:
                    fill = True
                    break
        if not fill:
            isrect = False
            break
    return isrect


# -------------------------------------------------
def colorful(rct, src):  # 物件內彩色像素計算
    ROI = cv2.cvtColor(src[rct[1]:rct[1] + rct[3], rct[0]:rct[0] + rct[2]], cv2.COLOR_BGR2HSV)
    count = 0
    for i in range(0, rct[3], 2):
        for j in range(0, rct[2], 2):
            if ROI[i, j, 1] > 50 and ROI[i, j, 2] > 20:
                count += 1
    return count > rct[2] * rct[3] * 0.06


# -------------------------------
def extract_table(images, src):  # 表格偵測演算法
    height, width = images.shape
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (int(width / 10), 1));
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(height / 55)));
    out_1d_hor = cv2.morphologyEx(images, cv2.MORPH_OPEN, horizontalStructure)
    out_1d_ver = cv2.morphologyEx(images, cv2.MORPH_OPEN, verticalStructure)
    mask = out_1d_hor + out_1d_ver
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, 2)
    _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lists = []
    for contour in contours:
        rct = cv2.boundingRect(contour)
        a = cv2.contourArea(contour, False)
        # if not colorful(rct,src):

        if a > 0.6 * rct[2] * rct[3]:
            if (rct[3] > height * 0.02 and rct[2] > width * 0.3) or (a > height * width * 0.05):
                lists.append(rct)
        else:
            maps = np.zeros([height, width], dtype=np.uint8)
            cv2.drawContours(maps, contour, -1, 255, 1)
            if findbox(rct, maps):
                if (rct[3] > height * 0.02 and rct[2] > width * 0.3):
                    lists.append(rct)
    return lists


# ----------------------------------
def isoverlap(lists, rct, threshold):  # 查看rct與list內的方框是否重疊
    for i in lists:
        x0 = max(i[0], rct[0])
        x1 = min(i[0] + i[2], rct[0] + rct[2])
        y0 = max(i[1], rct[1])
        y1 = min(i[1] + i[3], rct[1] + rct[3])
        if x0 >= x1 or y0 >= y1:
            continue
        else:
            result = (x1 - x0) * (y1 - y0) / ((i[2] * i[3]) + (rct[2] * rct[3]) - (x1 - x0) * (y1 - y0))
            if result > threshold:
                return True
    return False


# ---------------------------------
def extract_some_image(pproc, src, lists):  # 圖片偵測演算法
    height, width, _ = src.shape
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(pproc, cv2.MORPH_CLOSE, kernel, 2)
    _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    record = []
    for contour in contours:
        rct = cv2.boundingRect(contour)
        if rct[3] > height * 0.1 and rct[2] > width * 0.1:
            if not isoverlap(lists, rct, 0.9):
                if colorful(rct, src):
                    record.append(rct)
                elif not isoverlap(lists, rct, 0.8):
                    record.append(rct)

    kernel1 = np.ones((3, 3), np.uint8)
    mask1 = cv2.morphologyEx(pproc, cv2.MORPH_OPEN, kernel1, 2)
    _, contours, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 10000:
            rct = cv2.boundingRect(contour)
            if colorful(rct, src) and rct[3] > height * 0.1 and rct[2] > 0.1 * width:
                record.append(rct)
    return record


# -------------------------------------------------------------------
def multi_process_form_image(img_source, images):  # 對圖與表偵測執行多執行序
    src = cv2.imread('{}/{}'.format(img_source, images))
    pproc_outs = preprocess(src)
    lists = extract_table(pproc_outs, src)
    img_lists = extract_some_image(pproc_outs, src, lists)
    return lists, img_lists


# -----------------------------------------------------------------
def Form_DCT(indir, file_name, origin_img_dir):  # 圖片表格表格偵測多執行序
    make_pic(indir, file_name, origin_img_dir)
    img_source = '{}/{}/'.format(origin_img_dir, file_name.split('.')[0])
    page_form_lists = [None for name in os.listdir(img_source)]
    page_img_lists = [None for name in os.listdir(img_source)]

    with multiprocessing.Pool() as pool:
        lst = []
        for images in os.listdir(img_source):
            res1 = pool.apply_async(multi_process_form_image, (img_source, images))
            lst.append([res1, images])
        for i in lst:
            this_page = int(i[1].split('.')[0].split('_')[1]) - 1
            lists, img_lists = i[0].get()
            page_form_lists[this_page] = lists
            page_img_lists[this_page] = img_lists
    print('Exteact Form Done!')
    return page_form_lists, page_img_lists


# ---------------------------------------------
def htplusimage(pdf_data_path, pdf):  # 內文偵測
    print("Manage {} ....".format(pdf))
    fp = open('{}/{}'.format(pdf_data_path, pdf), 'rb')
    parser = PDFParser(fp)  # 創建文檔分析器
    document = PDFDocument(parser)  # 創建pdf對象除存文檔結構
    # name=pdf.split('.')[0]
    if not document.is_extractable:
        print('NOT EXTRACTABLE')
        raise PDFTextExtractionNotAllowed
    else:
        rsrcmgr = PDFResourceManager()  # 建立pdf資源管理器對象除存共享資源
        laparams = LAParams()  # 進行參數分析
        laparams.word_margin = 0.5
        device = PDFPageAggregator(rsrcmgr, laparams=laparams)  # 建立pdf設備對象
        interpreter = PDFPageInterpreter(rsrcmgr, device)  # 創見一個pdf解釋器對象

        big_head = []
        big_head_L = []
        big_tail_L = 0
        big_tail_C = 0
        page_rec_h = []
        page_rec_w = []
        all_record_txt = []
        all_record_image = []

        for page in PDFPage.create_pages(document):
            page_rec_h.append(page.mediabox[3])
            page_rec_w.append(page.mediabox[2])
            interpreter.process_page(page)
            layout = device.get_result()  # 接受此頁面的ltpage對象
            min_height = 10000
            all_rec_page_txt = []
            record_image = []
            for item in layout:  # 獲取頁首以及最尾端之文字位置
                if isinstance(item, LTTextBoxHorizontal):
                    tmp_h = item.get_text().replace("\n", "").replace(" ", "")
                    if tmp_h:
                        all_rec_page_txt.append(item)
                        maxxx = max(item.bbox[1], item.bbox[3])
                        minxx = min(item.bbox[1], item.bbox[3])
                        if minxx > page.mediabox[3] * 7 / 8:  # and len(tmp_h)<30:
                            big_head.append(tmp_h)
                            big_head_L.append(minxx)
                        elif maxxx < page.mediabox[3] / 8 and maxxx > 0.025 * page.mediabox[3] and len(tmp_h) < 30:
                            if maxxx < min_height:
                                min_height = maxxx
                elif isinstance(item, LTFigure) or isinstance(item, LTImage):
                    record_image.append(item)

            all_record_image.append(record_image)
            all_record_txt.append(all_rec_page_txt)

            if min_height < 10000:
                big_tail_L += min_height
                big_tail_C += 1

        [(top_hh, _)] = Counter(page_rec_h).most_common(1)  # find most common resolution height
        # Counter_dct=Counter(big_head)#統計每種頁首出現次數
        ccc = countsimilar(big_head)
        max_head_txt = []

        for i in ccc:  # find frequence of head
            if i[1] and i[1] > 5:
                max_head_txt.append(i[0])

        cnt = 0
        minus_avg_head = 0
        if max_head_txt:
            avg_head = 0
            for index, i in enumerate(big_head):
                for j in max_head_txt:
                    if i and SequenceMatcher(None, i, j).ratio() > 0.5:
                        # if (page_rec[index]-most_common)*2 > (page_rec[index]-big_head_L[index]):
                        avg_head += big_head_L[index]
                        cnt += 1
            minus_avg_head = top_hh - avg_head / cnt
        else:
            print('NO_head')

        avg_tail = big_tail_L / big_tail_C + 5
        pdf_txt = []
        pdf_txt_txt = []

        for idx, pagess in enumerate(all_record_txt):
            page_text = []
            page_txt_txt = []
            for items in pagess:
                if (items.bbox[1] > avg_tail and page_rec_h[idx] - items.bbox[3] > minus_avg_head) or abs(
                        items.bbox[1] - items.bbox[3]) > 0.05 * page_rec_h[idx]:
                    page_txt_txt.append(items.get_text())
                    if page_rec_w[idx] > items.bbox[2]:
                        page_text.append(
                            [items.bbox[0] / page_rec_w[idx], (page_rec_h[idx] - items.bbox[1]) / page_rec_h[idx],
                             items.bbox[2] / page_rec_w[idx], (page_rec_h[idx] - items.bbox[3]) / page_rec_h[idx]])
                    else:
                        page_text.append(
                            [items.bbox[0] / page_rec_w[idx], (page_rec_h[idx] - items.bbox[1]) / page_rec_h[idx], 1.0,
                             (page_rec_h[idx] - items.bbox[3]) / page_rec_h[idx]])

            pdf_txt.append(page_text)
            pdf_txt_txt.append(page_txt_txt)
        # ----------
        pdf_pic = []
        for idx, pagess in enumerate(all_record_image):
            page_pic = []
            for items in pagess:
                if abs((items.bbox[1] - items.bbox[3])) > 50 and abs((items.bbox[2] - items.bbox[0])) > 50:
                    page_pic.append(
                        [items.bbox[0] / page_rec_w[idx], (page_rec_h[idx] - items.bbox[1]) / page_rec_h[idx],
                         items.bbox[2] / page_rec_w[idx], (page_rec_h[idx] - items.bbox[3]) / page_rec_h[idx]])
            pdf_pic.append(page_pic)

        return pdf_txt, pdf_pic, pdf_txt_txt, minus_avg_head / top_hh, (top_hh - avg_tail) / top_hh


# ----------------------------------------
def checks(mask, rct, threshold):  # 物件重疊之評估
    cnts = 0
    maxx = max(rct[3], rct[1])
    minx = min(rct[3], rct[1])
    maxy = max(rct[0], rct[2])
    miny = min(rct[0], rct[2])
    for i in range(minx, maxx, 2):
        for j in range(miny, maxy, 2):
            if mask[i, j]:
                cnts += 1
    return cnts < (maxy - miny) * (maxx - minx) * threshold * 0.125


# -----
def checks1(mask, rct, threshold):  # 是否與mask重疊之評估
    cnts = 0
    maxx = max(rct[3], rct[1])
    minx = min(rct[3], rct[1])
    maxy = max(rct[0], rct[2])
    miny = min(rct[0], rct[2])
    if maxx > mask.shape[0]:
        maxx = mask.shape[0]
    if maxy > mask.shape[1]:
        maxy = mask.shape[1]
    tmp = None
    for i in range(minx, maxx, 2):
        for j in range(miny, maxy, 2):
            if mask[i, j]:
                tmp = mask[i, j]
                cnts += 1
    if tmp:
        return cnts < (maxy - miny) * (maxx - minx) * threshold * 0.125, tmp - 1
    else:
        return cnts < (maxy - miny) * (maxx - minx) * threshold * 0.125, None


# -----
def chk_all_v1(mask, pthdwn, pthup, w0, w2):  # find caption 搜索上下固定範圍
    POS = None
    NEG = None

    for i in range(w2 + 80, w0 - 20, -5):
        if 0 < i < mask.shape[1]:
            if not POS:
                for j in range(0, +100, 10):
                    if j + pthup < mask.shape[0]:
                        if mask[j + pthup, i]:
                            POS = [j, mask[j + pthup, i] - 1]
                            break
            if not NEG:
                for j in range(0, -250, -10):
                    if j + pthdwn > 0:
                        if mask[j + pthdwn, i]:
                            NEG = [j, mask[j + pthdwn, i] - 1]
                            break
        if POS and NEG:
            break
    if POS or NEG:
        return True, POS, NEG
    return False, None, None


# ---
def merge_horizontal_form_image(h, w, img_lst, f_img_lst, form_lst, dir1, dir3, name, tmp, thresh_h):  # 合併水平重疊之圖表
    boxes = []
    for i_box in img_lst:
        notmatch = True
        box_tmp = [int(i_box[0] * w), int(min(i_box[1], i_box[3]) * h), int(i_box[2] * w),
                   int(max(i_box[1], i_box[3]) * h)]
        if box_tmp[1] < thresh_h:
            continue
        for idx, box in enumerate(boxes):
            if min(box_tmp[2], box[2]) < max(box_tmp[0], box[0]) or min(box_tmp[3], box[3]) < max(box_tmp[1], box[1]):
                continue
            else:
                boxes[idx] = [min(box_tmp[0], box[0]), min(box_tmp[1], box[1]), max(box_tmp[2], box[2]),
                              max(box_tmp[3], box[3])]
                notmatch = False
                break
        if notmatch:
            boxes.append(box_tmp)

    flags = [0 for _ in boxes]

    for f_i_box in f_img_lst:
        notmatch = True
        box_tmp = [f_i_box[0], f_i_box[1], f_i_box[0] + f_i_box[2], f_i_box[1] + f_i_box[3]]
        # print(box_tmp)
        # if box_tmp[1]<thresh_h:
        #     continue
        for idx, box in enumerate(boxes):
            if min(box_tmp[2], box[2]) < max(box_tmp[0], box[0]) or min(box_tmp[3], box[3]) < max(box_tmp[1], box[1]):
                continue
            else:
                boxes[idx] = [min(box_tmp[0], box[0]), min(box_tmp[1], box[1]), max(box_tmp[2], box[2]),
                              max(box_tmp[3], box[3])]
                notmatch = False
                break
        if notmatch:
            boxes.append(box_tmp)
            flags.append(1)

    for f_box in form_lst:
        notmatch = True
        box_tmp = [f_box[0], f_box[1], f_box[0] + f_box[2], f_box[1] + f_box[3]]
        # if box_tmp[3]>thresh_h:
        #     continue
        for idx, box in enumerate(boxes):
            if min(box_tmp[2], box[2]) < max(box_tmp[0], box[0]) or min(box_tmp[3], box[3]) < max(box_tmp[1], box[1]):
                continue
            else:
                boxes[idx] = [min(box_tmp[0], box[0]), min(box_tmp[1], box[1]), max(box_tmp[2], box[2]),
                              max(box_tmp[3], box[3])]
                notmatch = False
                break
        if notmatch:
            boxes.append(box_tmp)
            flags.append(2)

    while 1:
        if len(boxes) > 1:
            ischange = False
            for idx, box in enumerate(boxes):
                for idx1, box1 in enumerate(boxes[idx + 1:]):
                    if min(box1[2], box[2]) < max(box1[0], box[0]) or min(box1[3], box[3]) < max(box1[1], box[1]):
                        continue
                    else:
                        boxes[idx] = [min(box[0], box1[0]), min(box[1], box1[1]), max(box[2], box1[2]),
                                      max(box[3], box1[3])]
                        # boxes.remove(box1)
                        del boxes[idx + idx1 + 1]
                        flags[idx] = min(flags[idx], flags[idx + idx1 + 1])
                        # flags.remove(flags[idx+idx1+1])
                        del flags[idx + idx1 + 1]
                        ischange = True
                        break
                if ischange:
                    break
            if not ischange:
                break
        else:
            break

    mask = np.zeros((h, w), np.uint8)
    mask_big = np.zeros((h, w), np.uint8)

    big_lst_name = []
    for idx, box in enumerate(boxes):
        if box[3] - box[1] < 0.1 * h:
            cv2.rectangle(mask_big, (box[0], box[1]), (box[2], box[3]), idx + 1, cv2.FILLED, 8)
        cv2.rectangle(mask, (box[0], box[1]), (box[2], box[3]), idx + 1, cv2.FILLED, 8)
        if flags[idx] < 2:
            big_lst_name.append("{}/{}/{}_{}_{}.jpg".format(dir3, name, name, tmp, idx))
        else:
            big_lst_name.append("{}/{}/{}_{}_{}.jpg".format(dir1, name, name, tmp, idx))
    return mask, mask_big, big_lst_name, boxes, flags


# ----
def clean_all_txt_in_item(page, boxes, new_filt_box, new_filt_txts):
    new_box = new_filt_box[:]
    new_txt = new_filt_txts[:]
    cnt = 0
    for idx, txt_box in enumerate(new_filt_box):
        isov = False
        for idx1, box in enumerate(boxes):
            if min(txt_box[2], box[2]) < max(txt_box[0], box[0]) or min(txt_box[3], box[3]) < max(txt_box[1], box[1]):
                continue
            else:
                isov = True
                boxes[idx1] = [min(box[0], txt_box[0]), min(box[1], txt_box[1]), max(box[2], txt_box[2]),
                               max(box[3], txt_box[3])]
                break
        if isov:
            del new_box[idx - cnt]
            del new_txt[idx - cnt]
            cnt += 1
    return boxes, new_box, new_txt


# -------------------------------------------------
def multi_draw(page_num, form_lst, f_img_lst, txt_lst, img_lst, val_txt_lst, origin_img_dir, name, all_dir, show_all,
               show_caption, thresh_h, thresh_t):  # 對所有物件 圖表內文做分析及篩選
    route = "{}/{}/{}".format(origin_img_dir, name, name)
    tmp = "%03d" % (page_num + 1)
    print("{}_{}.jpg".format(route, tmp))
    img = cv2.imread("{}_{}.jpg".format(route, tmp))
    h, w, _ = img.shape
    img1 = np.copy(img)
    # mask=np.zeros((h,w), np.uint8)
    mask, mask_big, big_lst_name, boxes, flags = merge_horizontal_form_image(h, w, img_lst, f_img_lst, form_lst,
                                                                             all_dir[0], all_dir[2], name, tmp,
                                                                             thresh_h)
    mask_with_txt = np.zeros((h, w), np.uint8)  # get filter txt and form image for non line

    title = ['' for _ in big_lst_name]  # 紀錄 每張圖片對應的 標題
    title_box = [None for _ in big_lst_name]  # 紀錄每張圖片的 標題位置
    tail = ['' for _ in big_lst_name]  # 紀錄每張圖片的 尾端文字
    tail_box = [None for _ in big_lst_name]  # 紀錄每張圖片的 尾端文字方塊位置
    every_page_min_height = h  # 紀錄文字塊區域的最高處
    all_txt_after_filt = []  # 紀錄濾除過後之文字位置
    all_txt_num = []  # 紀錄濾除過後之文字
    # ------------------------------這部份利用merge_horizontal_form_image 產生出來對於圖表的mask 進行caption的偵測以及濾除
    for idxs, txt_box in enumerate(txt_lst):
        # 物件之格式歸一 可能有1 <=> 3的狀況
        box = [int(min(txt_box[0], txt_box[2]) * w), int(min(txt_box[1], txt_box[3]) * h),
               int(max(txt_box[0], txt_box[2]) * w), int(max(txt_box[1], txt_box[3]) * h)]  # 1 small ,3 big
        if every_page_min_height > box[1]:
            every_page_min_height = box[1]
        isov, which = checks1(mask, (box[0], box[1], box[2], box[3]), 0.4)  # 查看是否已經與mask重疊
        if isov:  # 若不重疊
            xxx = ''.join(val_txt_lst[idxs].split())
            flag, POS, NEG = chk_all_v1(mask, box[1], box[3], box[0], box[2])  # 判斷文字上下範圍左右有無mask=1 若有代表可能為caption
            # 一次偵測pos 跟neg 若沒有偵測到則返回null 返回值[位置,第幾個物件]
            if flag and len(xxx):
                if POS:  # word beyond image
                    numb = POS[0]
                    TAGS = POS[1]
                    # ------
                    if numb < 11:
                        if (box[2] - box[0]) * (box[3] - box[1]) < 2 * w:
                            # cv2.rectangle(img1, (box[0], box[1]), (box[2],box[3]), (0,128,111), 3, 8)
                            if flags[TAGS] == 2:
                                if boxes[TAGS][3] - boxes[TAGS][1] > 0.1 * h:
                                    boxes[TAGS] = [min(boxes[TAGS][0], box[0]), min(boxes[TAGS][1], box[1]),
                                                   max(boxes[TAGS][2], box[2]), max(boxes[TAGS][3], box[3])]
                            else:
                                boxes[TAGS] = [min(boxes[TAGS][0], box[0]), min(boxes[TAGS][1], box[1]),
                                               max(boxes[TAGS][2], box[2]), max(boxes[TAGS][3], box[3])]
                            continue
                    # -------
                    if numb <= 80:
                        if xxx[0] == '图':
                            # if (txt_box[2]*w-txt_box[0]*w)<0.7*mask.shape[1]:
                            cv2.rectangle(img1, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 3, 8)
                            cv2.rectangle(mask_with_txt, (box[0], box[1]), (box[2], box[3]), 255, cv2.FILLED, 8)
                            title[TAGS] = xxx
                            title_box[TAGS] = box
                            continue
                    # -----
                    if numb <= 100:
                        if (xxx[0] == '表' and xxx[-1] != '。') or xxx[-1] == '表':
                            if (box[2] - box[0]) < 0.65 * w or numb < 21:
                                cv2.rectangle(img1, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 3, 8)
                                cv2.rectangle(mask_with_txt, (box[0], box[1]), (box[2], box[3]), 255, cv2.FILLED, 8)
                                title[TAGS] = xxx
                                title_box[TAGS] = box
                                continue

                        elif len(xxx) < 9 and xxx[:2] == '单位':  # 找單位開頭且字段小
                            cv2.rectangle(img1, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 3, 8)
                            cv2.rectangle(mask_with_txt, (box[0], box[1]), (box[2], box[3]), 255, cv2.FILLED, 8)
                            continue
                        elif (box[2] - box[0]) < 0.5 * w and 7 < len(xxx) < 50 and abs(
                                (box[0] + box[2] - w) / 2) < w / 15 and '。' not in xxx:  # 找附近置中文字段
                            cv2.rectangle(img1, (box[0], box[1]), (box[2], box[3]), (123, 255, 0), 3, 8)
                            cv2.rectangle(mask_with_txt, (box[0], box[1]), (box[2], box[3]), 255, cv2.FILLED, 8)
                            continue
                    if numb <= 20:
                        if '单位' in xxx:
                            cv2.rectangle(img1, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 3, 8)
                            cv2.rectangle(mask_with_txt, (box[0], box[1]), (box[2], box[3]), 255, cv2.FILLED, 8)
                            continue
                # -------
                if NEG:
                    numb = NEG[0]
                    TAGS = NEG[1]
                    # ------
                    if numb > -11:
                        if (box[2] - box[0]) * (box[3] - box[1]) < 2 * w:
                            cv2.rectangle(img1, (box[0], box[1]), (box[2], box[3]), (0, 128, 111), 3, 8)
                            if flags[TAGS] == 2:
                                if boxes[TAGS][3] - boxes[TAGS][1] > 0.1 * h:
                                    boxes[TAGS] = [min(boxes[TAGS][0], box[0]), min(boxes[TAGS][1], box[1]),
                                                   max(boxes[TAGS][2], box[2]), max(boxes[TAGS][3], box[3])]
                            else:
                                boxes[TAGS] = [min(boxes[TAGS][0], box[0]), min(boxes[TAGS][1], box[1]),
                                               max(boxes[TAGS][2], box[2]), max(boxes[TAGS][3], box[3])]
                            continue
                    if numb > -21:  # 較勁的可以叫松散
                        if '来源' in xxx[:7]:
                            cv2.rectangle(img1, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 3, 8)
                            cv2.rectangle(mask_with_txt, (box[0], box[1]), (box[2], box[3]), 255, cv2.FILLED, 8)
                            # caption_tmp.append([box[0],box[1]])
                            tail[TAGS] = xxx
                            tail_box[TAGS] = box
                            continue

                    if xxx[0] == '图':
                        if box[2] - box[0] < 0.7 * w:
                            cv2.rectangle(img1, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 3, 8)
                            cv2.rectangle(mask_with_txt, (box[0], box[1]), (box[2], box[3]), 255, cv2.FILLED, 8)
                            # title[TAGS]=xxx
                            continue
                    if numb >= -250:
                        if len(xxx) > 5 and box[2] - box[0] < 0.7 * w:
                            if '来源' in xxx[:7]:
                                cv2.rectangle(img1, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 3, 8)
                                cv2.rectangle(mask_with_txt, (box[0], box[1]), (box[2], box[3]), 255, cv2.FILLED, 8)
                                tail[TAGS] = xxx
                                tail_box[TAGS] = box
                                continue
                    if '注' in xxx[:1]:
                        cv2.rectangle(img1, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 3, 8)
                        cv2.rectangle(mask_with_txt, (box[0], box[1]), (box[2], box[3]), 255, cv2.FILLED, 8)
                        tail[TAGS] = xxx
                        tail_box[TAGS] = box
                        continue

            if len(xxx) < 10 and '单位：' in xxx:
                cv2.rectangle(img1, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 3, 8)
                cv2.rectangle(mask_with_txt, (box[0], box[1]), (box[2], box[3]), 255, cv2.FILLED, 8)
                continue

            all_txt_after_filt.append(box)
            all_txt_num.append(val_txt_lst[idxs])
        else:  # 若有文字與物件重疊 就把文字與物件合併擴張
            boxes[which] = [min(boxes[which][0], box[0]), min(boxes[which][1], box[1]), max(boxes[which][2], box[2]),
                            max(boxes[which][3], box[3])]
    # -----------------------------利用剛剛找初步的caption頭尾 來修補無法正確偵測的圖表
    for idx, box in enumerate(boxes):  # 針對caption首尾內對物件進行文字搜索擴張
        if title_box[idx] and tail_box[idx]:
            tmp_box = box[:]
            del_lst = []
            for idx1, txt in enumerate(list(all_txt_after_filt)):
                if title_box[idx][1] < txt[1] < tail_box[idx][1]:
                    if title_box[idx][2] - title_box[idx][0] < box[2] - box[0]:
                        if box[0] < (txt[0] + txt[2]) >> 1 < box[2]:
                            del_lst.append(idx1)
                            tmp_box = [min(txt[0], tmp_box[0]), min(txt[1], tmp_box[1]), max(txt[2], tmp_box[2]),
                                       max(txt[3], tmp_box[3])]
                    else:
                        if title_box[idx][0] < (txt[0] + txt[2]) >> 1 < title_box[idx][2]:
                            del_lst.append(idx1)
                            tmp_box = [min(txt[0], tmp_box[0]), min(txt[1], tmp_box[1]), max(txt[2], tmp_box[2]),
                                       max(txt[3], tmp_box[3])]
            for idx1, i in enumerate(del_lst):
                del all_txt_after_filt[i - idx1]
            boxes[idx] = tmp_box

            # #------這邊利用先前定義每頁的文字邊界進行區間定義 減少偵測失誤
    every_page_min_height -= 5
    # cv2.line(img1,(0,int(thresh_t*h)),(w,int(thresh_t*h)),(255,0,0),1)
    # cv2.line(img1,(0,int(thresh_h*h)),(w,int(thresh_h*h)),(255,0,0),1)
    # cv2.line(img1,(0,int(every_page_min_height)),(w,int(every_page_min_height)),(255,0,0),1)
    # #------------- 利用現有偵測結果之物件匡再做一個mask 目的是要給無格線表格用(覆蓋掉先前的mask)
    mask = np.zeros((h, w), np.uint8)
    for idx, box in enumerate(boxes):
        if box[3] - box[1] > 0.1 * h:
            cv2.rectangle(mask, (box[0], box[1]), (box[2], box[3]), idx + 1, cv2.FILLED, 8)
    # # # #--------------無格線表格的偵測
    non_line_boxes = get_nonline_table(img, mask_with_txt + mask, every_page_min_height, thresh_t)
    # non_line_boxes=mg_pdf_non(bbbbz,non_line_boxes0)
    merge_non_line_boxes = []
    for nlf_box in non_line_boxes:
        if (nlf_box[3] - nlf_box[1]) > 0.01 * h and checks(mask, nlf_box, 0.15):
            notmerge = True
            for box in merge_non_line_boxes:
                boxmg = non_line_ovlap(nlf_box, box)
                if boxmg:
                    notmerge = False
                    merge_non_line_boxes.remove(box)
                    merge_non_line_boxes.append(boxmg)
                    break
            if notmerge:
                merge_non_line_boxes.append(nlf_box)
    # ---------------------------- 由於新增了無格線表格至結果 因此也要真對於格線表格的caption進行律除
    mask_tmp_for_nonline = np.zeros((h, w), np.uint8)
    new_filt_txts = []  # filt out all of the title of nonline txt
    new_filt_box = []
    non_title = ['' for _ in merge_non_line_boxes]
    non_title_box = [None for _ in merge_non_line_boxes]
    non_tail_box = [None for _ in merge_non_line_boxes]
    non_big_name = []
    for idx, box in enumerate(merge_non_line_boxes):
        cv2.rectangle(mask_tmp_for_nonline, (box[0], box[1]), (box[2], box[3]), int(idx + 1), cv2.FILLED, 8)
        non_big_name.append("{}/{}/{}_{}_{}.jpg".format(all_dir[6], name, name, tmp, idx))
    for idx, box in enumerate(all_txt_after_filt):
        isov, _ = checks1(mask_tmp_for_nonline, (box[0], box[1], box[2], box[3]), 0.4)
        if isov:
            flag, POS, NEG = chk_all_v1(mask_tmp_for_nonline, box[1], box[3], box[0], box[2])
            xxx = ''.join(all_txt_num[idx].split())
            if flag and len(xxx):
                if POS:
                    numb = POS[0]
                    TAGS = POS[1]
                    if (xxx[0] == '图') or (xxx[0] == '表' and xxx[-1] != '。') or (xxx[-1] == '表'):
                        cv2.rectangle(img1, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 4, 8)
                        non_title[TAGS] = xxx
                        non_title_box[TAGS] = box
                        continue
                if NEG:
                    numb = NEG[0]
                    TAGS = NEG[1]
                    if '来源' in xxx[:5]:
                        non_tail_box[TAGS] = box
                        cv2.rectangle(img1, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 4, 8)
                        continue
            new_filt_txts.append(all_txt_num[idx])
            new_filt_box.append(all_txt_after_filt[idx])
    title.extend(non_title)
    title_box.extend(non_title_box)
    tail_box.extend(non_tail_box)
    big_lst_name.extend(non_big_name)
    boxes.extend(merge_non_line_boxes)
    flags.extend([3 for _ in merge_non_line_boxes])
    # ------
    bbbbz = use_miner_txt_nonline(img, page_num, new_filt_box)
    # for boxa in bbbbz:
    #     cv2.rectangle(img1, (boxa[0], boxa[1]), (boxa[2], boxa[3]), (255,0,255), 3, 8)
    boxes = mg_pdf_non(bbbbz, boxes, flags)
    boxes, new_filt_box, new_filt_txts = clean_all_txt_in_item(page_num, boxes, new_filt_box, new_filt_txts)
    # ----------------------------- 至此可能會有物件重疊之問題 因此需要合併物件
    while 1:
        if len(boxes) > 1:
            ischange = False
            for idx, box in enumerate(boxes):
                for idx1, box1 in enumerate(boxes[idx + 1:]):
                    if min(box1[2], box[2]) + 5 < max(box1[0], box[0]) or min(box1[3], box[3]) + 15 < max(box1[1],
                                                                                                          box[1]):
                        continue
                    else:
                        boxes[idx] = [min(box[0], box1[0]), min(box[1], box1[1]), max(box[2], box1[2]),
                                      max(box[3], box1[3])]
                        del boxes[idx + idx1 + 1]
                        if flags[idx] > flags[idx + idx1 + 1]:  # 以圖表為優先，相較於無格線表格 flag 1 2是圖表 ，無格現表格3
                            flags[idx] = flags[idx + idx1 + 1]
                            big_lst_name[idx] = big_lst_name[idx + idx1 + 1]

                        del flags[idx + idx1 + 1]

                        if title[idx] == '':
                            title[idx] = title[idx + idx1 + 1]
                        del title[idx + idx1 + 1]
                        del big_lst_name[idx + idx1 + 1]
                        ischange = True
                        break
                if ischange:
                    break
            if not ischange:
                break
        else:
            break
    # for boxa in new_filt_box:
    #     cv2.rectangle(img1, (boxa[0], boxa[1]), (boxa[2], boxa[3]), (255,0,255), 3, 8)
    # -----最後針對所有物件匡律除文字
    # mask=np.zeros((h,w), np.uint8)
    # for idx,box in enumerate(boxes):
    #     cv2.rectangle(mask, (box[0], box[1]), (box[2], box[3]), idx+1, cv2.FILLED, 8)
    # ----------
    # boxes,new_filt_box,new_filt_txts=clean_all_txt_in_item(page_num,boxes,new_filt_box,new_filt_txts)
    # -------------------------
    # txt output 文字輸出順便律除一些輸出上的問題
    with open("{}/{}/{}_{}.txt".format(all_dir[1], name, name, tmp), "w") as text_file:
        for idx, txt in enumerate(new_filt_txts):
            xxx = ''.join(txt.split())

            if xxx[0] == '图' and (new_filt_box[idx][2] - new_filt_box[idx][0]) < 0.6 * w and (
                    new_filt_box[idx][3] - new_filt_box[idx][1]) < 0.1 * h:
                cv2.rectangle(img1, (new_filt_box[idx][0], new_filt_box[idx][1]),
                              (new_filt_box[idx][2], new_filt_box[idx][3]), (0, 255, 0), 3, 8)
                continue

            if '来源' in xxx[:5] and (new_filt_box[idx][2] - new_filt_box[idx][0]) < 0.6 * w:
                cv2.rectangle(img1, (new_filt_box[idx][0], new_filt_box[idx][1]),
                              (new_filt_box[idx][2], new_filt_box[idx][3]), (0, 255, 0), 3, 8)
                continue

            txt = re.sub(r'\n \[(\w*?)\]', '', txt)
            txt = re.sub(r'．?', '', txt)
            text_file.write(txt)

            if show_all:
                cv2.rectangle(img1, (new_filt_box[idx][0], new_filt_box[idx][1]),
                              (new_filt_box[idx][2], new_filt_box[idx][3]), (0, 255, 255), 3, 8)

    # save image form 輸出可視化結果
    for idxs, box in enumerate(boxes):
        if flags[idxs] == 3:
            cv2.imwrite(big_lst_name[idxs], img[box[1]:box[3], box[0]:box[2]])
            if show_all:
                cv2.rectangle(img1, (box[0], box[1]), (box[2], box[3]), (128, 0, 128), 3, 8)
        elif flags[idxs] == 2:
            cv2.imwrite(big_lst_name[idxs], img[box[1]:box[3], box[0]:box[2]])
            if show_all:
                cv2.rectangle(img1, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 3, 8)
        else:
            cv2.imwrite(big_lst_name[idxs], img[box[1]:box[3], box[0]:box[2]])
            if show_all:
                cv2.rectangle(img1, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 3, 8)

    if show_all:
        cv2.imwrite("{}/{}/{}_{}.jpg".format(all_dir[3], name, name, tmp), img1)
    # --------------
    # with open('outfile/{}/{}_{}'.format(name,name,tmp), 'wb') as fp:
    #     pickle.dump([boxes,flags], fp)
    # -------------------
    return title, big_lst_name


# ----
def mg_pdf_non(list_pdf, list_non, flags):
    for idx, box in enumerate(list_pdf):
        for idx1, box1 in enumerate(list_non):
            if flags[idx1] != 1:
                if min(box1[2], box[2]) + 5 < max(box1[0], box[0]) or min(box1[3], box[3]) + 30 < max(box1[1], box[1]):
                    continue
                else:
                    print('wwwww~~~')
                    list_non[idx1] = [min(box[0], box1[0]), min(box[1], box1[1]), max(box[2], box1[2]),
                                      max(box[3], box1[3])]
                    break
    return list_non


def find_horizontal_group_for_pdfminer(box_list):
    list_tree = []
    flag = [True for _ in box_list]
    count = []
    for idx, box in enumerate(box_list):
        for idx1, node in enumerate(list_tree):
            if (box[3] - node[1]) * (box[1] - node[3]) <= 0:
                list_tree[idx1] = [min(box[0], node[0]), min(box[1], node[1]), max(box[2], node[2]),
                                   max(box[3], node[3])]
                flag[idx] = False
                count[idx1] = True
                break
        if flag[idx]:
            list_tree.append(box)
            flag[idx] = False
            count.append(False)
    new = [i for idx, i in enumerate(list_tree) if count[idx]]
    new1 = sorted(new, key=lambda x: x[1])
    return new1


def use_miner_txt_nonline(src, page_num, txt_lst):
    h, w, _ = src.shape
    list_1 = find_horizontal_group_for_pdfminer(txt_lst)
    list_new = []
    if list_1:
        while 1:
            ischange = False
            for idx, box in enumerate(list_1):
                for boxn in list_1[idx + 1:]:
                    if (box[1] - boxn[3]) * (box[3] - boxn[1]) < 0 or min(abs(box[1] - boxn[3]),
                                                                          abs(box[3] - boxn[1])) < 0.07 * h:
                        list_1[idx] = [min(box[0], boxn[0]), min(box[1], boxn[1]), max(box[2], boxn[2]),
                                       max(box[3], boxn[3])]
                        list_1.remove(boxn)
                        ischange = True
                        break
                if ischange:
                    break
            if not ischange:
                break
        for i in list_1:
            if i[3] - i[1] < 0.1 * h:
                list_new.append(i)
        return list_new
    return []


# ----
def non_line_ovlap(box1, box2):  # 另一個查看是否重疊
    minx1 = min(box1[0], box1[2])
    miny1 = min(box1[1], box1[3])
    maxx1 = max(box1[0], box1[2])
    maxy1 = max(box1[1], box1[3])
    minx2 = min(box2[0], box2[2])
    miny2 = min(box2[1], box2[3])
    maxx2 = max(box2[0], box2[2])
    maxy2 = max(box2[1], box2[3])
    if max(minx1, minx2) > min(maxx1, maxx2) or max(miny1, miny2) > min(maxy1, maxy2) + 20:
        return False
    else:
        return [min(minx1, minx2), min(miny1, miny2), max(maxx1, maxx2), max(maxy1, maxy2)]


# --------------------------------------------------
def drawbox_v1(form_lst, f_img_lst, txt_lst, img_lst, val_txt, origin_img_dir, name, all_dir, show_all, show_caption,
               thresh_h, thresh_t):  # 多執行序執行統整
    initial('{}/{}'.format(all_dir[0], name))
    initial('{}/{}'.format(all_dir[1], name))
    initial('{}/{}'.format(all_dir[2], name))
    initial('{}/{}'.format(all_dir[6], name))
    initial('{}/{}'.format(all_dir[7], name))
    if show_all:
        initial('{}/{}'.format(all_dir[3], name))
    with multiprocessing.pool.Pool() as pool:
        lst = []
        for page_num in range(len(form_lst)):
            res1 = pool.apply_async(multi_draw, (
            page_num, form_lst[page_num], f_img_lst[page_num], txt_lst[page_num], img_lst[page_num], val_txt[page_num],
            origin_img_dir, name, all_dir, show_all, show_caption, thresh_h, thresh_t))
            lst.append(res1)
        lst_name = []
        lst_caption = []
        for i in lst:
            a, b = i.get()
            lst_caption.extend(a)
            lst_name.extend(b)
        with open("{}/{}.txt".format(all_dir[5], name), "w") as text_file1:
            for itm in range(len(lst_caption)):
                titles = lst_caption[itm] if len(lst_caption[itm]) > 0 else 'No Title'
                text_file1.write("{}   {}\n".format(lst_name[itm], titles))


# --------------
def merge_txt(dir2, dir5, name, page_num):  # merge all page  context into one files
    with open("{}/{}.txt".format(dir5, name), "w") as all_text_file:
        for num in range(page_num):
            tmp = "%03d" % (num + 1)
            with open("{}/{}/{}_{}.txt".format(dir2, name, name, tmp), "r") as text_file:
                all_text_file.write(text_file.read())


#####################################################
def get_nonline_table(src, mask, thresh_h, thresh_t):  # get non-grid table
    height, width, _ = src.shape

    thresh_t *= height
    hsv_img = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    (_, _, V_img) = cv2.split(hsv_img)
    _, thresh1 = cv2.threshold(V_img, 25, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((1, width // 45), np.uint8)
    thresh2 = cv2.dilate(thresh1, kernel, iterations=1)
    _, contours, _ = cv2.findContours(thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    tmp_rct = []
    for contour in reversed(contours):
        rct = cv2.boundingRect(contour)
        if rct[2] > 15 and rct[1] > thresh_h and rct[1] + rct[3] < thresh_t:
            tmp_rct.append([rct[0], rct[1], rct[0] + rct[2], rct[1] + rct[3]])

    if tmp_rct:
        tmp_rct1 = []
        for box in tmp_rct:
            if checks(mask, box, 0.7):
                tmp_rct1.append(box)
        lstree_mg, list_org = mergev1(tmp_rct1)  # find horizontal group and filt out overlap part
        if len(lstree_mg):
            boxes = cluster3(height, width, lstree_mg, list_org)  # find similarities from preliminary result
            boxes1 = []
            for i in boxes:
                if i[3] - i[1] > 0.075 * height:
                    boxes1.append(i)
            return boxes1
    return []


# -----------
def find_horizontal_group(box_list):
    list_tree = []
    flag = [True for _ in box_list]
    for idx, box in enumerate(box_list):
        for idx1, nodes in enumerate(list_tree):
            for node in nodes:
                if (box[3] - node[1]) * (box[1] - node[3]) <= 0:
                    list_tree[idx1].append(box)
                    flag[idx] = False
                    break
            if not flag[idx]:
                break
        if flag[idx]:
            list_tree.append([box])
            flag[idx] = False
    sort_list_tree = []
    for nodes in list_tree:
        # if len(nodes)>1:
        sort_list_tree.append(sorted(nodes, key=lambda x: x[0]))
    return sort_list_tree


# ----
def mergev1(box_list):  # 水平聚類
    list_tree = find_horizontal_group(box_list)
    list_tree1 = []
    for nodes in list_tree:
        if len(nodes) > 1:
            now = nodes[0]
            filt_hor = []
            for node in nodes[1:]:
                if min(abs(node[2] - now[0]), abs(now[2] - node[0])) < max((node[3] - node[1]),
                                                                           (now[3] - now[1])) * 0.7:
                    now = [min(now[0], node[0]), min(now[1], node[1]), max(now[2], node[2]), max(now[3], node[3])]
                else:
                    filt_hor.append(now)
                    now = node
            filt_hor.append(now)
            if len(filt_hor) > 1:
                list_tree1.append(filt_hor)

    list_tree2 = real_mgovlap(list_tree1, True)
    list_tree_org = real_mgovlap(list_tree, False)
    return list_tree2, list_tree_org


# ----
def real_mgovlap(list_tree1, istree):  # 真正合併重疊
    list_tree2 = []
    for nodes in list_tree1:
        while 1:
            if len(nodes) > 1:
                all_not_change = True
                for idx, node in enumerate(nodes):
                    for node1 in nodes[idx + 1:]:
                        if min(node1[2], node[2]) < max(node1[0], node[0]) or min(node1[3], node[3]) < max(node1[1],
                                                                                                           node[1]):
                            continue
                        else:
                            nodes[idx] = [min(node1[0], node[0]), min(node1[1], node[1]), max(node1[2], node[2]),
                                          max(node1[3], node[3])]
                            nodes.remove(node1)
                            all_not_change = False
                            break
                    if not all_not_change:
                        break
                if all_not_change:
                    break
            else:
                break
        if istree:
            if len(nodes) > 1:
                list_tree2.append(nodes)
        else:
            list_tree2.extend(nodes)
    return list_tree2


# -----
def cluster3(h, w, box_list_tree, more_box_list):  # 無格線表格聚類
    lists = []
    listin = box_list_tree[0]
    for boxes in box_list_tree[1:]:
        if boxes[-1][3] - listin[-1][3] < 0.1 * h:
            listin.extend(boxes)
        else:
            lists.append(listin)
            listin = boxes
    lists.append(listin)

    boxes = []
    for group in lists:
        if len(group) > 3:
            new_gp = find_similarity(h, w, more_box_list, group)
            tmp = new_gp[0][:]
            for box in new_gp[1:]:
                if box[0] < tmp[0]:
                    tmp[0] = box[0]
                if box[1] < tmp[1]:
                    tmp[1] = box[1]
                if box[2] > tmp[2]:
                    tmp[2] = box[2]
                if box[3] > tmp[3]:
                    tmp[3] = box[3]
            boxes.append(tmp)
    return boxes


# ---
def find_similarity(h, w, more_box_list, less_box_list):  # 無格線表格聚類
    more1 = []
    for box in more_box_list:  # filt out overlap part
        notovlap = True
        for boxin in less_box_list:
            if min(box[2], boxin[2]) < max(box[0], boxin[0]) or min(box[3], boxin[3]) < max(box[1], boxin[1]):
                continue
            else:
                notovlap = False
                break
        if notovlap:
            more1.append(box)
    for box in more1:
        for boxin in less_box_list:
            if min(abs(box[1] - boxin[3]), abs(box[3] - boxin[1])) < 0.025 * h:
                if boxin[0] > box[2] or boxin[2] < box[0]:
                    continue
                elif (box[2] - box[0]) <= (boxin[2] - boxin[0]) * 1.2:
                    less_box_list.append(box)
                    break

    for box in more1:
        if box not in less_box_list:
            for boxin in less_box_list:
                if box[1] <= boxin[3] and box[3] >= boxin[1]:
                    less_box_list.append(box)
                    break
    return less_box_list


#########################
# --------------
def main(file_name, indir, origin_img_dir, all_dir, show_all, show_whole_txt, show_caption):
    tStart = time.time()
    t = ThreadWithReturnValue(target=Form_DCT, args=(indir, file_name, origin_img_dir))
    t.start()
    t1 = ThreadWithReturnValue(target=htplusimage, args=(indir, file_name))
    t1.start()
    val1, val2 = t.join()

    val3, val4, val_txt, thresh_h, thresh_t = t1.join()
    name = file_name.split('.')[0]
    drawbox_v1(val1, val2, val3, val4, val_txt, origin_img_dir, name, all_dir, show_all, show_caption, thresh_h,
               thresh_t)

    if show_whole_txt:
        merge_txt(all_dir[1], all_dir[4], name, len(val_txt))

    print('Finish {}'.format(file_name))
    tEnd = time.time()
    print("It cost %f sec" % (tEnd - tStart))  # 會自動做近位


# ------------------------------------------------
if __name__ == '__main__':
    indir = './pdf_data'
    origin_img_dir = './test'
    initial(origin_img_dir)

    all_dir = ['./FORM', './TXT', './IMAGE', './whole', './whole_txt', './caption', './nonline', './outfile']
    initial(all_dir[0])
    initial(all_dir[1])
    initial(all_dir[2])

    show_all = True
    if show_all:
        initial(all_dir[3])

    show_whole_txt = True
    if show_whole_txt:
        initial(all_dir[4])

    show_caption = True
    if show_caption:
        initial(all_dir[5])

    initial(all_dir[6])
    initial(all_dir[7])

    # all_dir=[dir1,dir2,dir3,dir4,dir5,dir6,dir7,dir8]

    # main('fku15.pdf',indir,origin_img_dir,all_dir,show_all,show_whole_txt,show_caption)

    consumer = KafkaConsumer('test')
    for msg in consumer:
        if str(msg.value).find('.pdf'):
            t0 = Thread(target=main,
                        args=(msg.value.decode('utf-8'), indir, origin_img_dir, dir1, dir2, dir3, dir4, show_all))
            t0.start()
    print('End')


