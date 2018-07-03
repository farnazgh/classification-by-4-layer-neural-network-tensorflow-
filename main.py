#!/usr/bin/env python


import tensorflow as tf
import numpy as np



# data preparation
# =================================================================

# load words polarity
words_polarity = {}
with open("data\Sentiment\EmotionLookupTable.txt", "r") as ins:
    for line in ins:
        items = line.split()
        word = items[0]
        word = word.replace('*','')

        score = int(items[1])

        words_polarity[word] = score


#load dataset
def line_to_features(line):

    words = line.split()

    total_pos_score = 0
    total_neg_score = 0

    count_pos_words = 0
    count_neg_words = 0

    for w in words:
        if w in words_polarity:
            score = words_polarity[w]
            if score>0:
                total_pos_score += score
                count_pos_words +=1
            else:
                total_neg_score += (-1*score)
                count_neg_words +=1

    return [total_pos_score, total_neg_score, count_pos_words, count_neg_words ]





X_pos=[]
Y_pos=[]
X_neg=[]
Y_neg=[]

with open("data\Sentiment\pos.txt", "r") as ins:
    for line in ins:
        X_pos.append(line_to_features(line));
        Y_pos.append([1,0]);


with open("data/Sentiment/neg.txt", "r") as ins:
    for line in ins:
        X_neg.append(line_to_features(line));
        Y_neg.append([0,1]);



index = int(len(Y_pos)*0.8)
x_train = np.array( X_pos[0:index] + X_neg[0:index] )
y_train = np.array( Y_pos[0:index] + Y_neg[0:index] )
x_test = np.array( X_pos[index:len(X_pos)] + X_neg[index:len(X_neg)] )
y_test = np.array( Y_pos[index:len(Y_pos)] + Y_neg[index:len(Y_neg)] )
y_test_cls = np.array([label.argmin() for label in y_test ])


print("Size of:")
print("- Training-set:\t\t{}".format(len(y_train)))
print("- Test-set:\t\t{}".format(len(y_test)))
print("- Test-set-cls:\t\t{}".format(len(y_test_cls)))








# model
# =================================================================

tf.reset_default_graph()

# num nodes
input_layer_nodes = 4
hidden_layer_nodes1 = 10
hidden_layer_nodes2 = 10
hidden_layer_nodes3 = 10
output_layer_nodes = 2


## place holders
x = tf.placeholder(tf.float32, [None, input_layer_nodes])
y_true = tf.placeholder(tf.float32, [None,output_layer_nodes])
y_true_cls = tf.placeholder(tf.int64, [None])


# hidden layer1

weights_hidden1 = tf.Variable(tf.random_normal([input_layer_nodes, hidden_layer_nodes1]))
bias_hidden1 = tf.Variable(tf.random_normal([hidden_layer_nodes1]))
preactivations_hidden1 = tf.add(tf.matmul(x, weights_hidden1), bias_hidden1)
activations_hidden1 = tf.nn.relu(preactivations_hidden1)

# hidden layer2

weights_hidden2 = tf.Variable(tf.random_normal([hidden_layer_nodes1, hidden_layer_nodes2]))
bias_hidden2 = tf.Variable(tf.random_normal([hidden_layer_nodes2]))
preactivations_hidden2 = tf.add(tf.matmul(activations_hidden1, weights_hidden2), bias_hidden2)
activations_hidden2 = tf.nn.relu(preactivations_hidden2)

# hidden layer3

weights_hidden3 = tf.Variable(tf.random_normal([hidden_layer_nodes2, hidden_layer_nodes3]))
bias_hidden3 = tf.Variable(tf.random_normal([hidden_layer_nodes3]))
preactivations_hidden3 = tf.add(tf.matmul(activations_hidden2, weights_hidden3), bias_hidden3)
activations_hidden3 = tf.nn.relu(preactivations_hidden3)

# output layer

weights_output = tf.Variable(tf.random_normal([hidden_layer_nodes3, output_layer_nodes]))
bias_output = tf.Variable(tf.random_normal([output_layer_nodes]))
logits = tf.add(tf.matmul(activations_hidden3, weights_output), bias_output)
y_pred = tf.nn.softmax(logits)
y_pred_cls = tf.argmin(y_pred, axis=1)


# cost function
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=y_true)
cost = tf.reduce_mean(cross_entropy)

# Optimization method
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

# Performance measures
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# session
session = tf.Session()
session.run(tf.global_variables_initializer())

## optimize helper function
batch_size = 100

def optimize():
    for j in range(int(len(x_train)/batch_size)):

        x_batch = x_train[batch_size*j:batch_size*(j+1)]
        y_true_batch = y_train[batch_size*j:batch_size*(j+1)]
       
        feed_dict_train = {x: x_batch, y_true: y_true_batch}

        session.run(optimizer, feed_dict=feed_dict_train)






# test data Performance
# =================================================================

feed_dict_test = {x: x_test, y_true: y_test, y_true_cls: y_test_cls}

def print_accuracy():
    acc = session.run(accuracy, feed_dict=feed_dict_test)
    print("Accuracy on test-set: {0:.1%}".format(acc))


num_iterations=60000
for i in range(num_iterations):
    print("iteration="+str(i))
    optimize()
    print_accuracy()  


