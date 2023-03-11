import pickle

import pandas as pd
from keras.preprocessing.text import Tokenizer
from konlpy.tag import Kkma
from sklearn.model_selection import train_test_split

# 데이터 입력
train = pd.read_csv('c:/csv/train_data.csv')

# 학습셋과 텍스트셋으로 나누기
x_train, x_test, y_train, y_test = train_test_split(train['passage'], train['doc_type'], test_size=0.2, random_state=11)

# 문서 타입별 번호
doc_type = {'briefing': 1, 'edit': 2, 'his_cul': 3, 'literature': 4, 'minute': 5, 'narration': 6, 'news_r': 7,
            'paper': 8, 'public': 9, 'speech': 10}

# 타입을 번호로 치환
y_train = y_train.replace(doc_type)
y_test = y_test.replace(doc_type)

# 문서 내용 데이터 전처리 하기
# 정규 표현식 수행
x_train = x_train.str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "")
x_test = x_test.str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "")

# 토큰화
train_tokens = []
test_tokens = []

for i in x_train:
    kkma = Kkma()
    tokens = kkma.morphs(i)
    train_tokens.append(tokens)

for i in x_test:
    kkma = Kkma()
    tokens = kkma.morphs(i)
    test_tokens.append(tokens)

# 불용어 제거
stop_words = pd.read_csv('C:/csv/stop_words.csv')
train_s = []
test_s = []

for i in tokens:
    if i not in stop_words:
        train_s.append(i)

for i in tokens:
    if i not in stop_words:
        test_s.append(i)

# 정수 인코딩
# 단어 집합 만들기
train_tokenizer = Tokenizer()
train_tokenizer.fit_on_texts(train_s)

test_tokenizer = Tokenizer()
test_tokenizer.fit_on_texts(test_s)

# 단어 집합 저장
with open('C:/pickle/train_tokenizer.pickle', 'wb') as f:
    pickle.dump(train_tokenizer, f)

# 등장 빈도수 계산후 낮은 빈도 단어 제거
rare_count = 3
train_total = len(train_tokenizer.word_index)
train_rare = 0
for key, value in train_tokenizer.word_counts.items():
    if value < rare_count:
        train_rare = train_rare + 1

train_maxlen = train_total - train_rare + 1
train_tokenizer = Tokenizer(train_maxlen)
train_tokenizer.fit_on_texts(x_train)
x_train = train_tokenizer.texts_to_sequences(x_train)

test_total = len(test_tokenizer.word_index)
test_rare = 0
for key, value in test_tokenizer.word_counts.items():
    if value < rare_count:
        test_rare = test_rare + 1

test_maxlen = test_total - test_rare + 1
test_tokenizer = Tokenizer(test_maxlen)
test_tokenizer.fit_on_texts(x_test)
x_test = test_tokenizer.texts_to_sequences(x_test)

# 전처리된 데이터 변수들 저장
with open('C:/pickle/pre_padding.pickle', 'wb') as f:
    pickle.dump(x_train, f)
    pickle.dump(x_test, f)
    pickle.dump(y_train, f)
    pickle.dump(y_test, f)
    pickle.dump(train_total, f)
    pickle.dump(train_rare, f)
