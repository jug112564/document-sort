import pickle

from keras.utils import pad_sequences, np_utils

with open('C:/pickle/pre_padding.pickle', 'rb') as f:
    x_train = pickle.load(f)
    x_test = pickle.load(f)
    y_train = pickle.load(f)
    y_test = pickle.load(f)
    train_total = pickle.load(f)
    train_rare = pickle.load(f)

print('최대 :', max(len(doc) for doc in x_train))

max_len = 70

# 길이 맞추기
x_train = pad_sequences(x_train, maxlen=max_len)
x_test = pad_sequences(x_test, maxlen=max_len)

# 원 핫 인코딩
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# 데이터 최종 변수 저장
with open('C:/pickle/data.pickle', 'wb') as f:
    pickle.dump(x_train, f)
    pickle.dump(x_test, f)
    pickle.dump(y_train, f)
    pickle.dump(y_test, f)
