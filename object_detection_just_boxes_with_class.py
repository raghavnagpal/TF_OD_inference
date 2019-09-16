import os
import datetime
import numpy as np
import cv2 as cv
import tensorflow as tf

MODEL_NAME = "/home/taira/work/downloaded_models/ssd_mobilenet_v1_coco_2018_01_28"
# MODEL_NAME = "/home/taira/work/downloaded_models/ssd_inception_v2_coco_2018_01_28"
#MODEL_NAME = "/home/taira/work/downloaded_models/mask_rcnn_inception_v2_coco_2018_01_28"
# MODEL_NAME = "/home/taira/work/downloaded_models/ssd_mobilenet_v2_coco_2018_03_29"
# MODEL_NAME = "/home/taira/work/downloaded_models/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28"
# MODEL_NAME = "/home/taira/work/downloaded_models/faster_rcnn_inception_v2_coco_2018_01_28"
# MODEL_NAME = "/home/taira/work/downloaded_models/faster_rcnn_resnet50_coco_2018_01_28"
# MODEL_NAME = "/home/taira/work/downloaded_models/faster_rcnn_resnet101_coco_2018_01_28"
# MODEL_NAME = "/home/taira/work/downloaded_models/mask_rcnn_inception_resnet_v2_atrous_coco_2018_01_28"

PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

class object_detection(object):
    def __init__(self,graph_path=''):
        if graph_path =='':
            print("pb graph path error")
            return

        self.graph_path = graph_path

        # graph
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            self.od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.graph_path, 'rb') as self.fid:
                self.serialized_graph = self.fid.read()
                self.od_graph_def.ParseFromString(self.serialized_graph)
                tf.import_graph_def(self.od_graph_def, name='')

        self.sess = tf.Session(graph=self.detection_graph)

        self.T_num_detections = self.detection_graph.get_tensor_by_name("num_detections:0")
        self.T_detection_boxes = self.detection_graph.get_tensor_by_name("detection_boxes:0")
        self.T_detection_scores = self.detection_graph.get_tensor_by_name("detection_scores:0")
        self.T_detection_classes = self.detection_graph.get_tensor_by_name("detection_classes:0")
        self.T_image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')

    def run(self,image = None):
        if image.all() == None:
            print("Empty image")
            return

        # self.h, self.w, _ = image.shape
        # invert
        self.image = image[::-1, :, :]

        self.image_np = self.image[:, :, [2, 1, 0]]
        self.image_np = np.expand_dims(self.image_np, axis=0)

        self.num_detections = self.T_num_detections
        self.detection_boxes = self.T_detection_boxes
        self.detection_scores = self.T_detection_scores
        self.detection_classes = self.T_detection_classes

        # Run inference
        (self.num_detections, self.detection_boxes, self.detection_scores, self.detection_classes) = self.sess.run(
            fetches=[self.num_detections, self.detection_boxes, self.detection_scores, self.detection_classes],
            feed_dict={self.T_image_tensor: self.image_np})
        
        return self.num_detections, self.detection_boxes, self.detection_scores, self.detection_classes

PATH_TO_TEST_IMAGES_DIR = '/home/taira/wpi/HIRO Lab/video analysis/samples/subj14_CA2_view/in'
imgnamelist = os.listdir(PATH_TO_TEST_IMAGES_DIR)
imgnamelist = np.sort(imgnamelist)

TEST_IMAGE_PATHS = [(os.path.join(PATH_TO_TEST_IMAGES_DIR, im)) for im in imgnamelist if im[-4:] in ['.jpg', '.png']]

OD = object_detection(PATH_TO_FROZEN_GRAPH)

for img_path in TEST_IMAGE_PATHS:

    tic = datetime.datetime.now()
    image = cv.imread(img_path)

    if image is None:
        continue
    h, w, _ = image.shape

    num_detections, detection_boxes, detection_scores, detection_classes = OD.run(image)
    # print(num_detections, detection_boxes, detection_scores, detection_classes)

    for i,box in enumerate(detection_boxes[0]):
        if detection_scores[0,i] > 0.2:
            # print(i,"box ",detection_boxes[0,i],detection_classes[0,i])

            image = cv.rectangle(image, (int(detection_boxes[0, i, 1] * w), int(detection_boxes[0, i, 0] * h)),
            (int(detection_boxes[0, i, 3] * w), int(detection_boxes[0, i, 2] * h)), (0, 255, 0), 3)

    print(img_path,"time per frame = ",datetime.datetime.now() - tic, )

    cv.imshow("image",image)
    cv.waitKey(1)
    cv.imwrite(img_path[:-4] + "_out_inv.jpg", image)

print("end")
