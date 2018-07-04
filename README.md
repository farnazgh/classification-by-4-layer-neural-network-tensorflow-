# Classification by 4 layer neural network (tensorflow)

Prediction task is to determine whether a person makes over 50K a year


Data: https://archive.ics.uci.edu/ml/datasets/Adult
	


four layer feedforwarding neural network : 

--input layer - 14 nodes (the total polarity of positive words, the total polarity of neg words, the total number of pos words, the total number of neg words) 

--hidden layers - the number of nodes in these layers regarding to the efficiency of network can be changed 

--output layer - 2 nodes 



features {
activation function: Relu,
cost function: cross entropy,
optimization method: Adam,
learning_rate=0.001,
batch size = 100,
iterations = 60000 }

accuracy = 80%
