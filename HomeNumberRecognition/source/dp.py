from __future__ import print_function,division
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import loadData
#load data
train_samples,train_labels=loadData.load("../data/train_32x32")
test_samples,test_labels=loadData.load("../data/test_32x32")

#data transformat
train_samples,train_labels=loadData.transformat(train_samples,train_labels)
test_samples,test_labels=loadData.transformat(test_samples,test_labels)

#normalized
train_samples=loadData.normalize(train_samples)
test_samples=loadData.normalize(test_samples)

print("the shape of train_set:",train_samples.shape)
print(train_samples[1:3])

image_size=train_samples.shape[1]
print ("image_size:",image_size)
num_labels=train_labels.shape[1]
print ("numlabels:",num_labels)
num_channels=train_samples.shape[3]
print ("numchannels",num_channels)





def get_chunk(samples, labels, chunkSize):
	'''
	Iterator/Generator: get a batch of data
	for loop just like range() function
	'''
	if len(samples) != len(labels):
		raise Exception('Length of samples and labels must equal')
	stepStart = 0	# initial step
	i = 0
	while stepStart < len(samples):
		stepEnd = stepStart + chunkSize
		if stepEnd < len(samples):
			yield i, samples[stepStart:stepEnd], labels[stepStart:stepEnd]
			i += 1
		stepStart = stepEnd

class Network():
    #init function
    def __init__(self,neurons_in_hidden,batch_size):
        """
        :param neurons_in_hidden:how many neurons in hidden layer
        :param batch_size:batch_size in SGD
        """

        self.batch_size = batch_size
        self.test_batch_size = 100
        #hyper parameters
        self.neurons_in_hidden=neurons_in_hidden

        #Graph related
        self.graph=tf.Graph();
        self.train_samples=None
        self.train_labels=None
        self.test_samples=None
        self.test_labels=None
        self.test_prediction=None


        self.define_graph()
        #define session
        self.session=tf.Session(graph=self.graph)

        #visualise
        writer=tf.train.SummaryWriter("./board",self.graph)


    #define_graph function
    def define_graph(self):
        with self.graph.as_default():
            '''
                define variables
            '''
            #data part
            self.train_samples=tf.placeholder(dtype=tf.float32,shape=(self.batch_size,image_size,image_size,num_channels))
            self.train_labels=tf.placeholder(dtype=tf.float32,shape=(self.batch_size,num_labels))
            self.test_samples=tf.placeholder(dtype=tf.float32,shape=(self.batch_size,image_size,image_size,num_channels))

            #fully connected layer 1(hidden layer)
            # weights(random) and biases(0.1)
            fc1_weights=tf.Variable(tf.random_normal(shape=(image_size*image_size,self.neurons_in_hidden),stddev=0.1))
            fc1_biases=tf.Variable(tf.constant(0.1,shape=[self.neurons_in_hidden]))

            #fully connected layer 2(output layer)
            # weights(random)and biases(0.1)
            fc2_weights=tf.Variable(tf.random_normal(shape=(self.neurons_in_hidden,num_labels),stddev=0.1))
            fc2_biases=tf.Variable(tf.constant(0.1,shape=[num_labels]))

            '''
                algorithm
            '''
            def model(data):
                #fully connect layer 1
                shape=data.get_shape().as_list()
                re_data=tf.reshape(data,shape=(shape[0],shape[1]*shape[2]*shape[3]))
                hidden=tf.nn.relu(tf.matmul(re_data,fc1_weights)+fc1_biases)

                #fully connect layer 2
                return tf.matmul(hidden,fc2_weights)+fc2_biases

            #loss
            logits=model(self.train_samples)
            self.loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits,self.train_labels))

            #optimizer
            self.optimizer=tf.train.GradientDescentOptimizer(0.0001).minimize(self.loss)

            #get the predictions for the training and test
            self.train_prediction=tf.nn.softmax(logits)
            self.test_prediction=tf.nn.softmax(model(self.test_samples))


   # def train(self):
    #    pass
    def run(self):
        # private function
        def print_confusion_matrix(confusionMatrix):
            print('Confusion    Matrix:')
            for i, line in enumerate(confusionMatrix):
                print(line, line[i] / np.sum(line))
            a = 0
            for i, column in enumerate(np.transpose(confusionMatrix, (1, 0))):
                a += (column[i] / np.sum(column)) * (np.sum(column) / 26000)
                print(column[i] / np.sum(column), )
            print('\n', np.sum(confusionMatrix), a)


        with self.session as sess:
            #initialize variables
            tf.initialize_all_variables().run()
            print ("Training:")

            #l:loss p:train_predictions
            for i,samples,labels in get_chunk(train_sacymples,train_labels,chunkSize=self.batch_size):
                _,l,p=sess.run(
                        [self.optimizer,self.loss,self.train_prediction],
                        feed_dict={self.train_samples:samples,self.train_labels:labels}
                )
                accuracy,_=self.accuracy(p,labels)
                if i%50==0:
                    print ("Minibatch loss at step %d:%f" %(i,l))
                    print ("Minibatch accuracy:%.1f%%"% accuracy)

            ### test
            accuracies = []
            confusionMatrices = []
            for i, samples, labels in get_chunk(test_samples, test_labels, chunkSize=self.test_batch_size):
                result = self.test_prediction.eval(feed_dict={self.test_samples: samples})
                accuracy, cm = self.accuracy(result, labels, need_confusion_matrix=True)
                accuracies.append(accuracy)
                confusionMatrices.append(cm)
                print('Test Accuracy: %.1f%%' % accuracy)
            print(' Average  Accuracy:', np.average(accuracies))
            print('Standard Deviation:', np.std(accuracies))
            print_confusion_matrix(np.add.reduce(confusionMatrices))
            ###



    def accuracy(self,predictions,labels,need_confusion_matrix=False):
        _predictions=np.argmax(predictions,1)
        _labels=np.argmax(labels,1)
        cm=confusion_matrix(_labels,_predictions) if need_confusion_matrix else None
        accuracy=(100.0*np.sum(_predictions==_labels)/predictions.shape[0])
        return accuracy,cm


if __name__=='__main__':
    net=Network(neurons_in_hidden=128,batch_size=100)
  #  net.define_graph()
    net.run()