# parser.py
from tqdm import tqdm
import json
import os
import requests
from bs4 import BeautifulSoup
from collections import OrderedDict
from algorithm.db import inser_chat_sequnece
from datetime import datetime


class Crawling:
    def __init__(self):
        url_list = ['http://www.diabetes.or.kr/general/class/index.php?idx=1',
                    'http://www.diabetes.or.kr/general/class/index.php?idx=2',
                    'http://www.diabetes.or.kr/general/class/index.php?idx=3',
                    'http://www.diabetes.or.kr/general/class/index.php?idx=4',
                    'http://www.diabetes.or.kr/general/class/index.php?idx=5',
                    'http://www.diabetes.or.kr/general/class/medical.php?mode=view&number=322&idx=6',
                    'http://www.diabetes.or.kr/general/class/medical.php?mode=view&number=325&idx=1',
                    'http://www.diabetes.or.kr/general/class/medical.php?mode=view&number=324&idx=1',
                    'http://www.diabetes.or.kr/general/class/medical.php?mode=view&number=323&idx=1',
                    'http://www.diabetes.or.kr/general/class/medical.php?mode=view&number=327&idx=2',
                    'http://www.diabetes.or.kr/general/class/medical.php?mode=view&number=326&idx=2',
                    'http://www.diabetes.or.kr/general/class/medical.php?mode=view&number=30&idx=4',
                    'http://www.diabetes.or.kr/general/class/medical.php?mode=view&number=7&idx=5',
                    'http://www.diabetes.or.kr/general/class/medical.php?mode=view&number=6&idx=5',
                    'http://www.diabetes.or.kr/general/class/complications.php?code=complication&number=337&mode=view&idx=1',
                    'http://www.diabetes.or.kr/general/class/complications.php?code=complication&number=336&mode=view&idx=2',
                    'http://www.diabetes.or.kr/general/class/type.php',
                    'http://www.diabetes.or.kr/general/class/gestational.php'
        ]

        file_count = len(os.walk('./json').__next__()[2]) + 1
        BASE_DIR = './json/'
        FILE_NAME = f'result{file_count}.json'

        json_arch = OrderedDict()
        json_arch["category"] = "당뇨병"

        title_list = []
        sub_title_list = []
        contents_list = []

        for url in tqdm(url_list):
            # HTTP GET Request
            req = requests.get(url)
            # HTML 소스 가져오기
            html = req.text
            # BeautifulSoup으로 html소스를 python객체로 변환하기
            # 첫 인자는 html소스코드, 두 번째 인자는 어떤 parser를 이용할지 명시.
            # 이 글에서는 Python 내장 html.parser를 이용했다.
            soup = BeautifulSoup(html, 'html.parser')

            # 타이틀
            title = soup.select_one('div.cTop > span:nth-of-type(2)')
            if title:
                title_list.append(title.text)
            else:
                title_list.append("")

            # 서브 타이틀
            all_sub_title = []
            content_all = soup.select_one('body')
            sub_title = soup.select('div.food')
            if sub_title:
                for sub_tit in content_all.find_all('div', 'rnd_center'):
                    all_sub_title.append(sub_tit.text)
                sub_title_list.append(all_sub_title)
            else:
                sub_title_list.append([])
            # 내용
            if len(sub_title) > 0:
                tmp_contents_list = []
                for idx in range(len(all_sub_title)):
                    tmp_contents = []
                    next_content = sub_title[idx].find_next_sibling('div')
                    while True:
                        next_content = next_content.find_next_sibling()
                        tmp_contents.append(next_content.text)
                        if next_content.find_next_sibling() == next_content.find_next_sibling('div') or next_content.find_next_sibling() == next_content.find_next_sibling('table'):
                            break
                    tmp_contents_list.append(tmp_contents)
                contents_list.append(tmp_contents_list)
            else:
                tmp_contents = []
                contents = soup.select('p.0')
                for content in contents:
                    tmp_contents.append(content.text)
                contents_list.append([tmp_contents])


        json_arch["title"] = title_list
        json_arch["sub_title"] = sub_title_list
        json_arch["content"] = contents_list

        with open(os.path.join(BASE_DIR, FILE_NAME), 'w', encoding="utf-8") as json_file:
            json.dump(json_arch, json_file, ensure_ascii=False, indent="\t")
        print("json 저장완료", "저장 경로", BASE_DIR, FILE_NAME)


class BuildDataSet:
    def __init__(self):
        LOAD_DIR = './json/result.json'
        self.question_list = []
        self.answer_list = []
        # os.walk('절대경로').next()[0] ==> 디렉토리 경로
        # os.walk('절대경로').next()[1] ==> 디렉토리 내의 디렉토리 개수
        # os.walk('절대경로').next()[2] ==> 디렉토리 내의 파일 개수
        with open(LOAD_DIR, 'rt', encoding='utf-8') as json_file:
            data = json.load(json_file)
            category = data["category"]
            for idx, tit in enumerate(data["title"]):
                if idx <= 7:
                    q = "%s" % tit
                else:
                    q = "%s %s" % (category, tit)
                print("질문:", q)
                self.question_list.append(q)
                if data["sub_title"][idx]:
                    a = "%s 관련 내용은 %s %d가지 종류가 있습니다. 어떤걸 알고 싶으신가요?" % (tit, data["sub_title"][idx], len(data["sub_title"][idx]))
                    print("내용:", a)
                    self.answer_list.append(a)
                    for idx2, sub_tit in enumerate(data["sub_title"][idx]):
                        q = sub_tit
                        self.question_list.append(q)
                        print("질문:", q)
                        a = self.down_vec(data["content"][idx][idx2])
                        self.answer_list.append(a)
                        print("내용:", a)
                else:
                    a = self.down_vec(data["content"][idx][0])
                    self.answer_list.append(a)
                    print("내용:", a)

    def save(self):
        data_set =[]
        for q, a in zip(self.question_list, self.answer_list):
            #print({"question": q, "answer": a, "pub_date": datetime.now()})
            data_set.append({"question": q, "answer": a, "pub_date": datetime.now()})
        inser_chat_sequnece(data_set)


    def down_vec(self, docs):
        temp = ""
        for doc in docs:
            #print(doc)
            temp += doc
        return temp


bds = BuildDataSet()
bds.save()

