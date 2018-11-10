import tensorflow as tf


# Seq2Seq 기본 클래스
class LogisticRegression:

    logits = None
    outputs = None
    cost = None
    train_op = None

    def __init__(self, x_data_size):
        self.learning_late = 0.001
        self.x_size = x_data_size

        self.X = tf.placeholder(tf.float32, [None, self.x_size])
        self.Y = tf.placeholder(tf.float32, [None, 1])

        self.weights = tf.Variable(tf.ones([self.x_size, 1]), name="weights")
        self.bias = tf.Variable(tf.zeros([1]), name="bias")

        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self._build_model()
        self.saver = tf.train.Saver(tf.global_variables())

    def _build_model(self):
        self.hypothesis, self.cost, self.train_op = self._build_ops()
        self.predicted = tf.cast(self.hypothesis > 0.5, dtype=tf.float32)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.predicted, self.Y), dtype=tf.float32))

    def _build_ops(self):
        # hypothesis = tf.sigmoid(tf.matmul(self.X, self.weights) + self.bias)
        # cost = -tf.reduce_mean(self.Y * tf.log(hypothesis) + (1 - self.Y) * tf.log(1 - hypothesis))
        # train_op = tf.train.GradientDescentOptimizer(learning_rate=self.learning_late).minimize(cost)
        hypothesis = tf.matmul(self.X, self.weights) + self.bias
        cost = tf.reduce_mean(tf.square(hypothesis - self.Y))
        train_op = tf.train.GradientDescentOptimizer(learning_rate=self.learning_late).minimize(cost)

        tf.summary.scalar('cost', cost)

        return hypothesis, cost, train_op

    def train(self, session, x_data, y_data):
        return session.run([self.train_op, self.cost],
                           feed_dict={self.X: x_data,
                                      self.Y: y_data})

    def test(self, session,  x_data, y_data):
        prediction_check = tf.cast(self.hypothesis > 0.5, dtype=tf.float32)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction_check, self.Y), dtype=tf.float32))

        return session.run([self.hypothesis, accuracy],
                           feed_dict={self.X: x_data,
                                      self.Y: y_data})

    def predict(self, session,  x_data, y_data):
        return session.run([self.predicted, self.accuracy],
                           feed_dict={self.X: x_data,
                                      self.Y: y_data})

    def write_logs(self, session, writer, x_data, y_data):
        merged = tf.summary.merge_all()
        summary = session.run(merged,
                              feed_dict={self.X: x_data,
                                         self.Y: y_data})

        writer.add_summary(summary, self.global_step.eval())