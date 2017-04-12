from __future__ import print_function,division
import numpy as np
import data
import tensorflow as tf

class Network():
    def __init__(self,num1,num2):
        # how many neurals in hidden layer,list
        self.num1=num1
        self.num2=num2

        #graph
        self.graph=tf.Graph()
        #define framework at the begining
        self.define_framwork()
        #session
        self.session=tf.Session(graph=self.graph)


    def define_framwork(self):
        '''
        #define framework of net
        :return:
        '''
        with self.graph.as_default():

            # fully connected layer1(connect input)
            # weights(random) and biases(0.1)
            with tf.name_scope("fc1"):
                self.fc1_weights=tf.Variable(initial_value=tf.random_normal(shape=(784, self.num1)),name="fc1_weights")
                self.fc1_biases=tf.constant(value=0.1,shape=(self.num1,),name="fc1_biases")

            # fully connected layer2
            # weights(random) and biases(0.1)
            with tf.name_scope("fc2"):
                self.fc2_weights=tf.Variable(initial_value=tf.random_normal(shape=(self.num1, self.num2)),name="fc1_weights")
                self.fc2_biases=tf.constant(value=0.1, shape=(self.num2,),name="fc2_biases")


            # fully connected layer n(output layer)
            # weights(random) and biases(0.1)
            with tf.name_scope("fc3"):
                self.fc3_weights=tf.Variable(initial_value=tf.random_normal(shape=(self.num2,10)),name="fc3_weights")
                self.fc3_biases=tf.constant(value=0.1,shape=(10,),name="fc3_biases")



    def forward(self,samples):
        '''
         # forward computation and return logits
        :param samples:(batch_size,784)
        :return:
        '''
        # we get batch_size x num1 matrix
        with tf.name_scope("fc1_hidden"):
            hidden_value1=tf.nn.relu(tf.matmul(samples,self.fc1_weights)+self.fc1_biases)
        with tf.name_scope("fc2_hidden"):
            hidden_value2 = tf.nn.relu(tf.matmul(hidden_value1, self.fc2_weights) + self.fc2_biases)

        # we get batch_size x 10 matrix(logits)
        with tf.name_scope("logits"):
            logits=tf.matmul(hidden_value2,self.fc3_weights)+self.fc3_biases
            return logits


    def train(self,epochs,train_batch_size,rate):
        '''
        #train net
        :param epochs:
        :param train_batch_size:
        :param rate:
        :return:
        '''
        #define graph

        with self.graph.as_default():
            # data part
            with tf.name_scope("input"):
                train_batch_samples = tf.placeholder(dtype=tf.float32, shape=(train_batch_size, 784),name="train_batch_samples")
                train_batch_labels = tf.placeholder(dtype=tf.float32, shape=(train_batch_size, 10),name="train_batch_labels")
            with tf.name_scope("train_logits"):
                train_batch_logits=self.forward(train_batch_samples)
            # train_loss
            # through softmax_...._logits,we get tensor of shape(batchs,)
            # through reduce_mean(),we get a "number"
            with tf.name_scope("loss"):
                train_batch_loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(train_batch_logits,train_batch_labels))
                tf.scalar_summary(tags="train_batch_loss",values=train_batch_loss)
            #optimizer
            with tf.name_scope("optimizer"):
                train_batch_optimizer=tf.train.GradientDescentOptimizer(learning_rate=rate).minimize(train_batch_loss)

            #prediction
            with tf.name_scope("prediction"):
                train_batch_prediction=tf.nn.softmax(train_batch_logits)

            #summary
            summary=tf.merge_all_summaries()

        # visulization
        writer = tf.train.SummaryWriter(logdir="./log", graph=self.graph)

        #run
        with self.session as sess:
            tf.initialize_all_variables().run()
            print ("----------Training Start----------")

            #epoch
            epoch=1
            while epoch<epochs:
                print ("epoch:",epoch)
                samples,labels=data.shuffle()

                #mini_batch
                for i in range(0,data.train_data_size,train_batch_size):
                    _,loss,prediction,summaries=sess.run(
                            fetches=[train_batch_optimizer,train_batch_loss,train_batch_prediction,summary],
                            feed_dict={train_batch_samples:samples[i:i+train_batch_size],train_batch_labels:labels[i:i+train_batch_size]}
                        )

                    print ("mini_batch",i,"~",i+train_batch_size,"of",epoch,"epochs")
                    print ("loss:",loss)
                    print ("accuracy:",self.accuracy(prediction,labels[i:i+train_batch_size]),"/",train_batch_size)

                    #add summary
                    writer.add_summary(summary=summaries,global_step=i)
                epoch+=1



    def validation(self):
        pass
    def test(self):
        pass
        #with self.graph.as_default():
            # data part
            #self.test_batch_samples = tf.placeholder(dtype=tf.float32, shape=(self.batch_size, 784))


    def accuracy(self,predictions,labels):
        _predictions=np.argmax(predictions,axis=1)
        _labels=np.argmax(labels,axis=1)
        accu=np.sum(_predictions==_labels)
        return accu
