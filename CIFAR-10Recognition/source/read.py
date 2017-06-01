import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

graph=tf.Graph()
session=tf.Session(graph=graph)

with graph.as_default():
    files=tf.train.match_filenames_once(pattern="../data/TFRecords/train.tfrecords").initialized_value()
    print("files:",files)
    filename_queue=tf.train.string_input_producer(string_tensor=files,shuffle=False)

    reader=tf.TFRecordReader()
    _,serialized_example=reader.read(queue=filename_queue)

    features=tf.parse_single_example(
            serialized=serialized_example,
            features={
                "img_raw": tf.FixedLenFeature([],tf.string),
                "label": tf.FixedLenFeature([],tf.int64),
                "width": tf.FixedLenFeature([], tf.int64)
            }
        )

    #get single feature
    raw = features["img_raw"]
    label = features["label"]
    width = features["width"]
    # decode raw,and out_type should be the same with original pic type
    image = tf.decode_raw(bytes=raw, out_type=tf.float32)

    #dont forget reshape!!!
    image_shaped=tf.reshape(tensor=image,shape=(32,32,3))
    init_op=tf.global_variables_initializer()

with session.as_default():
    session.run(init_op)
    print(session.run(files))

    #threads
    coord=tf.train.Coordinator()
    threads=tf.train.start_queue_runners(sess=session,coord=coord)

    image_run,label_run=session.run(fetches=[image_shaped,label])
    print(type(image_run))
    print(image_run.shape)
    print(image_run)
    print(label_run)

    plt.imshow(image_run)
    plt.show()
    coord.request_stop()
    coord.join(threads)
