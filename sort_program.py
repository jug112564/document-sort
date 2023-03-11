import re

import numpy as np
from konlpy.tag import Kkma
import pandas as pd
from keras.utils import pad_sequences
import pickle
from keras.models import load_model


def sort_doc(doc):
    # 정규표현식
    doc = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣 ]', '', doc)

    # 토큰화
    kkma = Kkma()
    doc = kkma.morphs(doc)

    # 불용어제거
    stopwords = pd.read_csv('C:/csv/stop_words.csv')
    doc = [word for word in doc if not word in stopwords]

    # 정수 인코딩
    with open('c:/pickle/train_tokenizer.pickle', 'rb') as w:
        train_tokenizer = pickle.load(w)
    encoded = train_tokenizer.texts_to_sequences([doc])

    # 패딩
    pad = pad_sequences(encoded, maxlen=70)
    # 예측
    model = load_model('sort_model.h5')
    predict_x = model.predict(pad)
    classes_x = np.argmax(predict_x, axis=1)

    doc_type = ['보도자료', '사설', '역사기록물&문화재', '문학', '회의록', '나레이션', '뉴스기사', '보고서', '간행물',
                '연설문']
    print('이 문서는 ', doc_type[classes_x[0] - 1], '입니다.')


f1 = open('C:/doc/news1.txt', 'rt', encoding='UTF8')
doc1 = f1.read()
sort_doc(doc1)

f2 = open('C:/doc/news2.txt', 'rt', encoding='UTF8')
doc2 = f2.read()
sort_doc(doc2)

f3 = open('C:/doc/news3.txt', 'rt', encoding='UTF8')
doc3 = f3.read()
sort_doc(doc3)