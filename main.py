#!/usr/bin/env python


import tensorflow as tf
import numpy as np



# data preparation
# =================================================================

age = 0
workclass = {"?":0, "Private":1, "Self-emp-not-inc":2, "Self-emp-inc":3, "Federal-gov":4, "Local-gov":5, "State-gov":6, "Without-pay":7, "Never-worked":8}
fnlwgt = 0
education = {"?":0,"Bachelors":1, "Some-college":2, "11th":3, "HS-grad":4, "Prof-school":5, "Assoc-acdm":6, "Assoc-voc":7, "9th":8, "7th-8th":9, "12th":10, "Masters":11, "1st-4th":12, "10th":13, "Doctorate":14, "5th-6th":15, "Preschool":16}
education_num = 0
marital_status = {"?":0,"Married-civ-spouse":1, "Divorced":2, "Never-married":3, "Separated":4, "Widowed":5, "Married-spouse-absent":6, "Married-AF-spouse":7}
occupation = {"?":0,"Tech-support":1, "Craft-repair":2, "Other-service":3, "Sales":4, "Exec-managerial":5, "Prof-specialty":6, "Handlers-cleaners":7, "Machine-op-inspct":8, "Adm-clerical":9, "Farming-fishing":10, "Transport-moving":11, "Priv-house-serv":12, "Protective-serv":13, "Armed-Forces":14}
relationship = {"?":0,"?":0,"Wife":1, "Own-child":2, "Husband":3, "Not-in-family":4, "Other-relative":5, "Unmarried":6}
race = {"?":0,"White":1, "Asian-Pac-Islander":2, "Amer-Indian-Eskimo":3, "Other":4, "Black":5}
sex = {"?":0,"Female":1, "Male":2}
capital_gain= 0
capital_loss = 0
hours_per_week = 0
native_country ={"?":0,"United-States":1, "Cambodia":2, "England":3, "Puerto-Rico":4, "Canada":5, "Germany":6, "Outlying-US(Guam-USVI-etc)":7, "India":8, "Japan":9, "Greece":10, "South":11, "China":12, "Cuba":13, "Iran":14, "Honduras":15, "Philippines":16, "Italy":17, "Poland":18, "Jamaica":19, "Vietnam":20, "Mexico":21, "Portugal":22, "Ireland":23, "France":24, "Dominican-Republic":25, "Laos":26, "Ecuador":27, "Taiwan":28, "Haiti":29, "Columbia":30, "Hungary":31, "Guatemala":32, "Nicaragua":33, "Scotland":34, "Thailand":35, "Yugoslavia":36, "El-Salvador":37, "Trinadad&Tobago":38, "Peru":39, "Hong":40, "Holand-Netherlands":41}
# income = {"<=50K":0, ">50K":1}

X = []
Y = []
with open("data\income\data.txt", "r") as ins:
    for line in ins:
        l = []
        line = line.replace("\n","")
        items = line.split(", ")


        l.append(int(items[0]) if items[0]!="?" else 0) #age
        l.append(workclass.get(items[1])) #workclass
        l.append(int(items[2]) if items[2]!="?" else 0) #fnlwgt
        l.append(education.get(items[3])) #education
        l.append(int(items[4]) if items[4]!="?" else 0) #education_num
        l.append(marital_status.get(items[5])) #marital_status
        l.append(occupation.get(items[6])) #occupation
        l.append(relationship.get(items[7])) #relationship
        l.append(race.get(items[8])) #race
        l.append(sex.get(items[9])) #sex
        l.append(int(items[10]) if items[10]!="?" else 0) #capital_gain
        l.append(int(items[11]) if items[11]!="?" else 0) #capital_loss
        l.append(int(items[12]) if items[12]!="?" else 0) #hours_per_week
        l.append(native_country.get(items[13])) #native_country
        
        X.append(l)
        Y.append([1,0] if items[14]=="<=50K" else [0,1])


index = int(len(Y)*0.8)
x_train = np.array( X[0:index])
y_train = np.array( Y[0:index])
x_test = np.array( X[index:len(X)])
y_test = np.array( Y[index:len(Y)])
y_test_cls = np.array([label.argmin() for label in y_test ])


print("Size of:")
print("- Training-set:\t\t{}".format(len(y_train)))
print("- Test-set:\t\t{}".format(len(y_test)))
print("- Test-set-cls:\t\t{}".format(len(y_test_cls)))

# model
# =================================================================

tf.reset_default_graph()

# num nodes
input_layer_nodes = 14
hidden_layer_nodes1 = 7
# hidden_layer_nodes2 = 4
hidden_layer_nodes3 = 3
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

# weights_hidden2 = tf.Variable(tf.random_normal([hidden_layer_nodes1, hidden_layer_nodes2]))
# bias_hidden2 = tf.Variable(tf.random_normal([hidden_layer_nodes2]))
# preactivations_hidden2 = tf.add(tf.matmul(activations_hidden1, weights_hidden2), bias_hidden2)
# activations_hidden2 = tf.nn.relu(preactivations_hidden2)

# hidden layer3

weights_hidden3 = tf.Variable(tf.random_normal([hidden_layer_nodes1, hidden_layer_nodes3]))
bias_hidden3 = tf.Variable(tf.random_normal([hidden_layer_nodes3]))
preactivations_hidden3 = tf.add(tf.matmul(activations_hidden1, weights_hidden3), bias_hidden3)
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


