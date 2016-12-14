import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Dataset https://codeonweb.com/entry/c1fa46e4-6cd4-42fe-8d56-36ec7826a6f1loading
mnist = input_data.read_data_sets("./samples/MNIST_data/", one_hot=True)

# Set up model
x = tf.placeholder(tf.float32, [None, 784]) #�� ������ ���� �Ҽ������� �̷���� 2���� ����.
W = tf.Variable(tf.zeros([784, 10])) #��ȣ�ۿ��ϴ� �۾� �׷��� ���� �������� ������ ����.
b = tf.Variable(tf.zeros([10]))  #��� �������� ���ǰų� ������ ����� �� ����. 784���� �̹��� ���͸� ���Ͽ� 10���� ������ ���Ÿ� ����.
y = tf.nn.softmax(tf.matmul(x, W) + b) #����!! matmul�Լ��� �̿��� x�� W�� ����ϰ� softmax����.

y_ = tf.placeholder(tf.float32, [None, 10]) #������ �Է��ϱ� ���� �� placeholder �߰�.

cross_entropy = -tf.reduce_sum(y_*tf.log(y)) #������Ʈ������ ���� // reduce.sum�� ������ �����Ҹ� ����.
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# Session
init = tf.initialize_all_variables()   #������ ���� �ʱ�ȭ

sess = tf.Session()
sess.run(init)

# Learning
for i in range(1000):  #õ�� �н� �ݺ�!! �ݺ��ܰ踶�� �н���Ʈ�κ��� ������ ������ 100���� �ϰ�ó��.
  batch_xs, batch_ys = mnist.train.next_batch(100)   #�ű⿡ train_step �ǵ�����.
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# Validation // ������ ������ �˷��ִ� argmax�Լ��� ����� �´� ���� �����ߴ��� Ȯ��!
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Result should be approximately 91%.  // �׽�Ʈ ��Ȯ�� Ȯ��.
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))