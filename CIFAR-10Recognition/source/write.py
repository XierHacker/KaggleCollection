import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf
import pandas as pd
import os


#get the amount of files in folder
def sizeOfFolder(folder_path):
    fileNameList = os.listdir(path=folder_path)
    size = 0
    for fileName in fileNameList:
        if (os.path.isfile(path=os.path.join(folder_path, fileName))):
            size += 1
    return size


#path(path of folder)
#if isTrain=True,labels can't be None
def pics_to_TFRecord(folder_path,labels=None,isTrain=False):
    size=sizeOfFolder(folder_path=folder_path)

    #train set
    if isTrain:
        if labels is None:
            print("labels can't be None!!!")
            return None
        if labels.shape[0]!=size:
            print("something wrong with shape!!!")
            return None
        writer=tf.python_io.TFRecordWriter("../data/TFRecords/train.tfrecords")
        for i in range(1,size+1):
            print("----------processing the ",i,"\'th image----------")
            filename=folder_path+str(i)+".png"
            img=mpimg.imread(fname=filename)
            width=img.shape[0]
            print(width)
            #trans to string
            img_raw=img.tostring()

            example=tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "img_raw":tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                        "label":tf.train.Feature(int64_list=tf.train.Int64List(value=[labels[i-1]])),
                        "width":tf.train.Feature(int64_list=tf.train.Int64List(value=[width]))
                    }
                )

            )
            writer.write(record=example.SerializeToString())
        writer.close()

    #test set
    else:
        writer = tf.python_io.TFRecordWriter("../data/TFRecords/test.tfrecords")
        for i in range(1, size + 1):
            print("----------processing the ", i, "\'th image----------")
            filename = folder_path + str(i) + ".png"
            img = mpimg.imread(fname=filename)
            width = img.shape[0]
            print(width)
            # trans to string
            img_raw = img.tostring()

            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "img_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                        "width": tf.train.Feature(int64_list=tf.train.Int64List(value=[width]))
                    }
                )

            )
            writer.write(record=example.SerializeToString())
        writer.close()





train_labels_frame=pd.read_csv("../data/trainLabels.csv")
train_labels_frame_dummy=pd.get_dummies(data=train_labels_frame)
#print(train_labels_frame_dummy)
train_labels_frame_dummy.pop(item="id")
#print(train_labels_frame_dummy)
train_labels_values_dummy=train_labels_frame_dummy.values
#print(train_labels_values_dummy)
train_labels_values=np.argmax(train_labels_values_dummy,axis=1)
#print(train_labels_values)


#write train record
pics_to_TFRecord(folder_path="../data/train/",labels=train_labels_values,isTrain=True)

#write test record
pics_to_TFRecord(folder_path="../data/test/")







'''
def pics_to_csv(path,isTrain=False):
    pass


'''