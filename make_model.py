import pickle

from keras.layers import Dense, LSTM, Embedding, Dropout
from keras.models import Sequential

# 데이터 입력
with open('C:/pickle/data.pickle', 'rb') as f:
    x_train = pickle.load(f)
    x_test = pickle.load(f)
    y_train = pickle.load(f)
    y_test = pickle.load(f)

# 모델 설정
model = Sequential()
model.add(Embedding(70, 100))
model.add(LSTM(100, activation='tanh'))
model.add(Dense(11, activation='softmax'))

# 모델 컴파일
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# 모델 실행
history = model.fit(x_train, y_train, batch_size=100, epochs=15, validation_data=(x_test, y_test))
# 테스트 정확도 출력
print('\n Test Accuracy: %.4f' % (model.evaluate(x_test, y_test)[1]))

# 모델 저장
model.save('./sort_model.h5')
