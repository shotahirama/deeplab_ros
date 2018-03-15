#!/usr/bin/env python
# coding: utf8

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import tensorflow as tf
import rospkg
import urllib
import os
import PIL.Image
import numpy as np
from copy import deepcopy
import get_dataset_colormap
import tarfile

class DeepLabROS(object):
    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = 513
    MODEL_URL ='http://download.tensorflow.org/models/deeplabv3_pascal_train_aug_2018_01_04.tar.gz'
    def __init__(self):

        self.graphpb_path = os.path.join("/tmp","deeplabv3_pascal_trainval","frozen_inference_graph.pb")
        self.model_path = os.path.join(rospkg.RosPack().get_path("deeplab_ros"),"model","deeplab_model.tar.gz")
        if not os.path.exists(self.model_path):
            rospy.loginfo("Model Download...")
            urllib.urlretrieve(self.MODEL_URL,self.model_path)
            rospy.loginfo("Success Downloading")

        graph=tf.Graph()
        graph_def = None
        tar_file = tarfile.open(self.model_path)
        for tar_info in tar_file.getmembers():
            if "frozen_inference_graph" in os.path.basename(tar_info.name):
                file_handle = tar_file.extractfile(tar_info)
                graph_def = tf.GraphDef.FromString(file_handle.read())
                break
        tar_file.close()
        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')
        with graph.as_default():
            tf.import_graph_def(graph_def, name="")
        config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
        self.sess = tf.Session(graph=graph, config=config)
        rospy.loginfo("Success import %s"%self.graphpb_path)
        self.imagemsg = None
        self.cvbridge = CvBridge()
        self.pub = rospy.Publisher("deeplab/image_raw",Image, queue_size=1)
        self.sub = rospy.Subscriber("/image_raw",Image,self.image_callback,queue_size=1)

    def image_callback(self, msg):
        self.imagemsg = deepcopy(msg)

    def segmentation(self):
        if not self.imagemsg:
            return
        msg = deepcopy(self.imagemsg)
        try:
            img = self.cvbridge.imgmsg_to_cv2(msg,"bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)
            return

        image = PIL.Image.fromarray(img[:, :, ::-1].copy())
        width, height = image.size
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = image.convert('RGB').resize(target_size, PIL.Image.ANTIALIAS)
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
        seg_map = batch_seg_map[0]
        seg_image = get_dataset_colormap.label_to_color_image(
            seg_map, get_dataset_colormap.get_pascal_name()).astype(np.uint8)

        self.pub.publish(self.cvbridge.cv2_to_imgmsg(seg_image,"rgb8"))


if __name__ == '__main__':
    rospy.init_node("deeplab_ros")
    dlr = DeepLabROS()
    r = rospy.Rate(100)
    while not rospy.is_shutdown():
        dlr.segmentation()
        r.sleep()
