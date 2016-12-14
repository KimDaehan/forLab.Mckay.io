import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Dataset https://codeonweb.com/entry/c1fa46e4-6cd4-42fe-8d56-36ec7826a6f1loading
mnist = input_data.read_data_sets("./samples/MNIST_data/", one_hot=True)

# Set up model
x = tf.placeholder(tf.float32, [None, 784]) #이 형태의 부정 소숫점으로 이루어진 2차원 덴서.
W = tf.Variable(tf.zeros([784, 10])) #상호작용하는 작업 그래프 간의 유지변경 가능한 덴서.
b = tf.Variable(tf.zeros([10]))  #계산 과정에서 사용되거나 심지어 변경될 수 있음. 784차원 이미지 벡터를 곱하여 10차원 벡터의 증거를 만듬.
y = tf.nn.softmax(tf.matmul(x, W) + b) #구현!! matmul함수를 이용해 x와 W를 계산하고 softmax입힘.

y_ = tf.placeholder(tf.float32, [None, 10]) #정답을 입력하기 위한 새 placeholder 추가.

cross_entropy = -tf.reduce_sum(y_*tf.log(y)) #교차엔트로피의 구현 // reduce.sum은 덴서의 모든원소를 더함.
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# Session
init = tf.initialize_all_variables()   #세션의 변수 초기화

sess = tf.Session()
sess.run(init)

# Learning
for i in range(1000):  #천번 학습 반복!! 반복단계마다 학습세트로부터 무작위 데이터 100개를 일괄처리.
  batch_xs, batch_ys = mnist.train.next_batch(100)   #거기에 train_step 피딩실행.
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# Validation // 원소의 색인을 알려주는 argmax함수를 사용해 맞는 라벨을 예측했는지 확인!
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Result should be approximately 91%.  // 테스트 정확도 확인.
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))