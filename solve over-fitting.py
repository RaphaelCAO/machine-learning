import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt

# load data
digits = load_digits()
X = digits.data
y = digits.target
y = LabelBinarizer().fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)


def add_layer(inputs, in_size, out_size, layer_name, activation_function=None, ):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, )
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    # here to dropout
    Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b, )
    return outputs


# define placeholder for inputs to network
keep_prob = tf.placeholder(tf.float32)
xs = tf.placeholder(tf.float32, [None, 64])  # 8x8
ys = tf.placeholder(tf.float32, [None, 10])

l1 = add_layer(xs,64,50,'first_layer',activation_function = tf.nn.tanh)
prediction = add_layer(l1,50,10,'result',activation_function = tf.nn.softmax)

cross_entropy = tf.reduce_mean( - tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

loss_train = []
loss_test = []
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(1000):
    sess.run(train_step,feed_dict = {xs:X_train,ys:y_train,keep_prob:0.3})

    loss_train.append(sess.run(cross_entropy,feed_dict = {xs:X_train,ys:y_train,keep_prob:1}))
    loss_test.append(sess.run(cross_entropy,feed_dict = {xs:X_test,ys:y_test,keep_prob:1}))
    
plt.figure("loss comparation")
plt.plot(range(len(loss_train)),loss_train,color = 'r',label="train")
plt.plot(range(len(loss_test)),loss_test,color='b',label="test")
plt.legend()
plt.show()
