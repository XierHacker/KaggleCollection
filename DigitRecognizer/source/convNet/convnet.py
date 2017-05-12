from __future__ import print_function,division
import numpy as np
import data
import tensorflow as tf


class Network():
    def __init__(self):

        #graph
        self.graph=tf.Graph()
        #define framework at the begining
        self.define_framwork()
        #session
        self.session=tf.Session(graph=self.graph)

        #visulization
      #  writer=tf.train.SummaryWriter(logdir="./log",graph=self.graph)

    def define_framwork(self):
        '''
        #define framework of net
        :return:
        '''
        with self.graph.as_default():
            # equals to filter parameter
            def make_conv_parameter(filter_height, filter_width, in_channels, out_channels):
                weights = tf.Variable(
                    initial_value=tf.random_normal(shape=(filter_height, filter_width, in_channels, out_channels)))
                biases = tf.Variable(initial_value=tf.constant(value=0.1, shape=(out_channels,)))
                return weights, biases

            def make_fc_parameter(in_num, out_num):
                # weights(random) and biases(0.1)
                weights = tf.Variable(initial_value=tf.random_normal(shape=(in_num, out_num)))
                biases = tf.constant(value=0.1, shape=(out_num,))
                return weights, biases
            #conv layers
            self.conv1_weights, self.conv1_biases = make_conv_parameter(3, 3, 1,5)
            self.conv2_weights, self.conv2_biases = make_conv_parameter(3, 3, 5,7)
            self.conv3_weights, self.conv3_biases = make_conv_parameter(3, 3, 7,9)
            self.conv4_weights, self.conv4_biases = make_conv_parameter(3, 3, 9,11)


            #fully connected layer
            self.fc1_weights,self.fc1_biases=make_fc_parameter(7*7*11,100)
            self.fc2_weights, self.fc2_biases = make_fc_parameter(100, 10)



    def forward(self,samples):
        '''
         # forward computation and return logits
        :param samples:shape(batch_size,28,28,1)
        :return:
        '''
        with self.graph.as_default():
            #through conv1,we get (batch_size,28,28,5)
            conv1=tf.nn.conv2d(samples,self.conv1_weights,strides=[1,1,1,1],padding="SAME")
            hidden_value1=tf.nn.relu(conv1+self.conv1_biases)

            # through conv2,we get (batch_size,28,28,7)
            conv2 = tf.nn.conv2d(hidden_value1, self.conv2_weights, strides=[1, 1, 1, 1], padding="SAME")
            hidden_value2 = tf.nn.relu(conv2 + self.conv2_biases)

            #pooling,we get [batch,14,14,7]
            hidden_value2= tf.nn.max_pool(
                hidden_value2,
                ksize=[1,2,2,1],
                strides=[1,2,2,1],
                padding="SAME"
            )


            # through conv3,we get (batch_size,14,14,9)
            conv3 = tf.nn.conv2d(hidden_value2, self.conv3_weights, strides=[1, 1, 1, 1], padding="SAME")
            hidden_value3 = tf.nn.relu(conv3 + self.conv3_biases)

            # through conv2,we get (batch_size,28,28,7)
            conv4 = tf.nn.conv2d(hidden_value3, self.conv4_weights, strides=[1, 1, 1, 1], padding="SAME")
            hidden_value4 = tf.nn.relu(conv4 + self.conv4_biases)

            # pooling,we get [batch,7,7,11]
            hidden_value4 = tf.nn.max_pool(
                hidden_value4,
                ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1],
                padding="SAME"
            )

            #through fully connectd layer
            shape=hidden_value4.get_shape().as_list()
            hidden_value4=tf.reshape(hidden_value4,shape=(shape[0],shape[1]*shape[2]*shape[3]))

            # we get batch_size x 100 matrix
            fc_hidden1=tf.nn.relu(tf.matmul(hidden_value4,self.fc1_weights)+self.fc1_biases)
            # we get batch_size x 10 matrix(logits)
            logits=tf.matmul(fc_hidden1,self.fc2_weights)+self.fc2_biases
            return logits


    def train(self,epochs,train_batch_size,rate):
        '''
        #train net
        :param epochs:
        :param train_batch_size:
        :param rate:
        :return:
        '''
        with self.graph.as_default():
            # data part
            train_batch_samples = tf.placeholder(dtype=tf.float32, shape=(train_batch_size,28,28,1))
            train_batch_labels = tf.placeholder(dtype=tf.float32, shape=(train_batch_size, 10))

            train_batch_logits=self.forward(train_batch_samples)
            # train_loss
            # through softmax_...._logits,we get tensor of shape(batchs,)
            # through reduce_mean(),we get a "number"
            train_batch_loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=train_batch_labels,logits=train_batch_logits))

            #optimizer
            train_batch_optimizer=tf.train.AdamOptimizer(learning_rate=rate).minimize(train_batch_loss)
           # train_batch_optimizer=tf.train.GradientDescentOptimizer(learning_rate=rate).minimize(train_batch_loss)

            #prediction
            train_batch_prediction=tf.nn.softmax(train_batch_logits)

        #run
        with self.session as sess:
            tf.global_variables_initializer().run()
            print ("----------Training Start----------")

            #epoch
            epoch=1
            while epoch<epochs:
                print ("epoch:",epoch)
                samples,labels=data.shuffle()

                #mini_batch
                for i in range(0,data.train_data_size,train_batch_size):
                    _,loss,prediction=sess.run(
                            fetches=[train_batch_optimizer,train_batch_loss,train_batch_prediction],
                            feed_dict={train_batch_samples:samples[i:i+train_batch_size],train_batch_labels:labels[i:i+train_batch_size]}
                        )

                    print ("mini_batch",i,"~",i+train_batch_size,"of",epoch,"epochs")
                    print ("loss:",loss)
                    print ("accuracy:",self.accuracy(prediction,labels[i:i+train_batch_size]),"/",train_batch_size)
                epoch+=1

        # visulization
      #  writer = tf.train.SummaryWriter(logdir="./log", graph=self.graph)

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