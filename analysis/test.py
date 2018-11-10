import tensorflow as tf
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt

# 랜덤에 의해 똑같은 결과를 재현하도록 시드 설정
tf.set_random_seed(777)

# 하이퍼파라미터
timesteps = seq_length = 7  # 1개 시퀀스의 길이(시계열데이터 입력 개수)
data_dim = 3  # 입력데이터의 컬럼 개수(Variable 개수)
output_dim = 1  # 결과데이터의 컬럼 개수
hidden_dim = 20  # 각 셀의 (hidden)출력 크기
keep_prob = 0.9 # dropout할 때 keep할 비율
forget_bias = 1.0
num_stacked_layers = 1

epoch_num = 10000  # 에폭 횟수(학습용전체데이터를 몇 회 반복해서 학습할 것인가 입력)
learning_rate = 0.01  # 학습률

# 데이터를 로딩한다.
stock_file_name =  '/chatbot/bmireport.csv'  # bmi 기록 데이터 파일
encoding = 'utf-8'
names = ['Stature', 'Weight', 'Bmi', 'Date']
raw_dataframe = pd.read_csv(stock_file_name, names=names, encoding=encoding)  # 판다스이용 csv파일 로딩
raw_dataframe.info()  # 데이터 정보 출력

# 데이터 확인
print(raw_dataframe)

# 시간열 데이터 제거
del raw_dataframe["Date"]

#==========================
x = np.array(raw_dataframe)
y = x[:, [-1]]
dataX = []
dataY = []
for i in range(0, len(y) - seq_length):
    _x = x[i:i + seq_length]
    _y = y[i + seq_length]
    print(_x, "->", _y)
    dataX.append(_x)
    dataY.append(_y)

train_size = int(len(dataY) * 0.7)
test_size = len(dataY) - train_size
trainX, testX = np.array(dataX[0:train_size]), np.array(dataX[train_size:len(dataX)])
trainY, testY = np.array(dataY[0:train_size]), np.array(dataY[train_size:len(dataY)])

X = tf.placeholder(tf.float32, [None, seq_length, data_dim])
Y = tf.placeholder(tf.float32, [None, 1])

targets = tf.placeholder(tf.float32, [None, 1])
predictions = tf.placeholder(tf.float32, [None, 1])

def lstm_cell():
    # LSTM셀을 생성
    # num_units: 각 Cell 출력 크기
    # forget_bias:  to the biases of the forget gate
    #              (default: 1)  in order to reduce the scale of forgetting in the beginning of the training.
    # state_is_tuple: True ==> accepted and returned states are 2-tuples of the c_state and m_state.
    # state_is_tuple: False ==> they are concatenated along the column axis.
    cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim,
                                        forget_bias=forget_bias, state_is_tuple=True, activation=tf.nn.softsign)
    if keep_prob < 1.0:
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
    return cell


stackedRNNs = [lstm_cell() for _ in range(num_stacked_layers)]
multi_cells = tf.contrib.rnn.MultiRNNCell(stackedRNNs, state_is_tuple=True) if num_stacked_layers > 1 else lstm_cell()

hypothesis, _state = tf.nn.dynamic_rnn(multi_cells, X, dtype=tf.float32)
hypothesis = tf.contrib.layers.fully_connected(hypothesis[:, -1], output_dim, activation_fn=tf.identity)

loss = tf.reduce_sum(tf.square(hypothesis - Y))
optimizer = tf.train.AdadeltaOptimizer(learning_rate)
train = optimizer.minimize(loss)

rmse = tf.sqrt(tf.reduce_mean(tf.squared_difference(targets, predictions)))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(epoch_num):
    _, l = sess.run([train, loss], feed_dict={X: trainX, Y: trainY})
    print(i, l)

testPredict = sess.run(hypothesis, feed_dict={X: testX})

import matplotlib.pyplot as plt
plt.plot(testY)
plt.plot(testPredict)
plt.show()
