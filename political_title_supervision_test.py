# -*- coding: utf-8 -*-

from Try_demo.political_title_supervision.political_title_supervision import Political_Title_Supervision
import io

def load_zc_title():
    
    zc_titles = []
    with io.open('./title/zc_title.txt', "r", encoding='utf-8') as f:
        while True:
            line = f.readline()
            if len(line.strip()) > 0:
                zc_titles.append(line.strip())   
            else:
                break
    print('zc_titles: ', len(zc_titles))
    zc_titles = set(zc_titles)
    return zc_titles



if __name__ == '__main__':
    
     title = '[董事会]>东方银星:关于聘任董事会秘书的公告'
     
     
     p = Political_Title_Supervision()
     print(p.f(title))
     '''
     zc_titles = load_zc_title()
     for title in zc_titles:
         if not p.f(title):
             print(title)
     '''

