import os
import datetime
import numpy as np
import cv2 as cv
import tensorflow as tf

# MODEL_NAME = "/home/taira/work/downloaded_models/ssd_mobilenet_v1_coco_2018_01_28"
# MODEL_NAME = "/home/taira/work/downloaded_models/ssd_inception_v2_coco_2018_01_28"
MODEL_NAME = "/home/taira/work/downloaded_models/mask_rcnn_inception_v2_coco_2018_01_28"
# MODEL_NAME = "/home/taira/work/downloaded_models/ssd_mobilenet_v2_coco_2018_03_29"
# MODEL_NAME = "/home/taira/work/downloaded_models/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28"
# MODEL_NAME = "/home/taira/work/downloaded_models/faster_rcnn_inception_v2_coco_2018_01_28"
# MODEL_NAME = "/home/taira/work/downloaded_models/faster_rcnn_resnet50_coco_2018_01_28"
# MODEL_NAME = "/home/taira/work/downloaded_models/faster_rcnn_resnet101_coco_2018_01_28"
# MODEL_NAME = "/home/taira/work/downloaded_models/mask_rcnn_inception_resnet_v2_atrous_coco_2018_01_28"

PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

# graph
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

PATH_TO_TEST_IMAGES_DIR = '/home/taira/wpi/HIRO Lab/inference/imgs_8919/img_640'
imgnamelist = os.listdir(PATH_TO_TEST_IMAGES_DIR)
imgnamelist = np.sort(imgnamelist)

TEST_IMAGE_PATHS = [(os.path.join(PATH_TO_TEST_IMAGES_DIR, im)) for im in imgnamelist if im[-4:] in ['.jpg', '.png']]

with detection_graph.as_default():
    with tf.Session() as sess:

        # T_num_detections = tf.get_default_graph().get_tensor_by_name("num_detections:0")
        # T_detection_boxes = tf.get_default_graph().get_tensor_by_name("detection_boxes:0")
        # T_detection_scores = tf.get_default_graph().get_tensor_by_name("detection_scores:0")
        # T_detection_classes = tf.get_default_graph().get_tensor_by_name("detection_classes:0")
        # T_image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

        T_num_detections = detection_graph.get_tensor_by_name("num_detections:0")
        T_detection_boxes = detection_graph.get_tensor_by_name("detection_boxes:0")
        T_detection_scores = detection_graph.get_tensor_by_name("detection_scores:0")
        T_detection_classes = detection_graph.get_tensor_by_name("detection_classes:0")
        T_image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

        for img_path in TEST_IMAGE_PATHS:

            tic = datetime.datetime.now()
            image = cv.imread(img_path)

            if image is None:
                continue
            h, w, _ = image.shape


            image_np = image[:, :, [2, 1, 0]]
            image_np = np.expand_dims(image_np, axis=0)

            num_detections = T_num_detections
            detection_boxes = T_detection_boxes
            detection_scores = T_detection_scores
            detection_classes = T_detection_classes

            # Run inference
            (num_detections, detection_boxes, detection_scores, detection_classes) = sess.run(
                fetches=[num_detections, detection_boxes, detection_scores, detection_classes],
                feed_dict={T_image_tensor: image_np})

            # print(num_detections, detection_boxes, detection_scores, detection_classes)

            for i,box in enumerate(detection_boxes[0]):
                if detection_scores[0,i] > 0.1:
                    # print(i,"box ",detection_boxes[0,i],detection_classes[0,i])

                    image = cv.rectangle(image, (int(detection_boxes[0, i, 1] * w), int(detection_boxes[0, i, 0] * h)),
                    (int(detection_boxes[0, i, 3] * w), int(detection_boxes[0, i, 2] * h)), (0, 255, 0), 3)

            print("time per frame = ",datetime.datetime.now() - tic, )

            cv.imshow("image",image)
            cv.waitKey(1)

print("end")