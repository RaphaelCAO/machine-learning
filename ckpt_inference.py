# -*- coding:utf-8 -*-
import functools
import sys, cv2
import numpy as np
import tensorflow as tf
import re

sys.path.append('/media/caoqi/HD2/PycharmProjects/bl/models/research')
sys.path.append('/media/caoqi/HD2/PycharmProjects/bl/models/research/slim')

from object_detection.builders import model_builder
from object_detection.utils import config_util

tf.logging.set_verbosity(tf.logging.INFO)


class Model(object):
    def __init__(self, checkpoint_dir, pipeline_config_path, label_map_path,
                 img_height_width=(1080, 1920)):
        self.colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 255, 0),
                       (255, 0, 255), (0, 165, 255), (147, 20, 255), (35, 142, 107), (255, 112, 132)]

        with open(label_map_path) as f:
            text = str(f.readlines())
            pattern = r"name:\s?\'(\w*)\'"
            self.label_names = re.findall(pattern, text)

        self.label_map_path = label_map_path
        model = self.build_mdodel(pipeline_config_path)

        # 建立input-output
        self.image_height, self.image_width = img_height_width
        self.x = tf.placeholder(dtype=tf.float32, shape=[1, self.image_height, self.image_width, 3])
        preprocessed_image, true_image_shapes = model.preprocess(tf.to_float(self.x))
        prediction_dict = model.predict(preprocessed_image, true_image_shapes)

        # 后处理过程中会对detection_scores添加激励函数，比如(sigmoid等)，因此用grad-cam不能用postprocess之后的scores值
        self.detections, self.all_scores_no_bg = model.postprocess(prediction_dict, true_image_shapes)

        saver = tf.train.Saver()
        self.sess = tf.Session()

        self.sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            print('No checkpoint found!')

    def run_inference_for_single_image(self, image):
        """
        inference one image
        :param image: image path or image numpy
        :return: the prediction result
        """
        if type(image) is str:
            image_np = cv2.imread(image)
        else:
            image_np = image

        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        image_np = np.expand_dims(image_np, axis=0)
        detections, all_scores_no_bg = self.sess.run([self.detections, self.all_scores_no_bg], feed_dict={self.x: image_np})

        num_detections = int(detections.get('num_detections').astype(int))
        detection_scores = detections.get('detection_scores').squeeze()
        detection_boxes = detections.get('detection_boxes').squeeze()[0:num_detections, :]
        detection_classes = detections.get('detection_classes').squeeze().astype(int)
        return detection_boxes, detection_scores, detection_classes

    def visualize_prediction(self, image, score_thres=0.3, window_name='window'):
        """
        Given a picture path, visualize the bounding boxes of predicted objects.
        :param img: the path of an image or image numpy
        :param score_thres: the score threshold for visible bounding box
        :return: None
        """
        if type(image) == str:
            image_np = cv2.imread(image)
        else:
            image_np = image
        boxes, scores, labels = self.run_inference_for_single_image(image)

        height, width, _ = image_np.shape
        for score, box, label in zip(scores, boxes, labels):
            if score < score_thres:
                break
            score = int(score * 100) / 100
            pt1 = (int(box[1] * width), int(box[0] * height))
            pt2 = (int(box[3] * width), int(box[2] * height))
            color = self.colors[label - 1]
            cv2.rectangle(image_np, pt1, pt2, color, 2)

            title = "{}: {}".format(self.label_names[label - 1], score)
            bar_width = np.ceil(score * (pt2[0] - pt1[0])).astype(np.int)
            cv2.rectangle(image_np, (pt1[0], pt1[1] - 35), (pt2[0], pt1[1]), color, 2)
            cv2.rectangle(image_np, (pt1[0], pt1[1] - 35), (pt1[0] + bar_width, pt1[1]), color, cv2.FILLED)
            cv2.putText(image_np, title,
                        (pt1[0], pt1[1] - 10), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)

        cv2.namedWindow(window_name, 0)
        cv2.imshow(window_name, image_np)
        return image_np

    def build_mdodel(self, pipeline_config_path):
        configs = config_util.get_configs_from_pipeline_file(pipeline_config_path)
        model_config = configs['model']

        model_fn = functools.partial(model_builder.build, model_config=model_config, is_training=False)

        # 建立graph,流程参考evaluator.py中的_extract_predictions_and_losses(),主要调用ssd_mata_arch.py中的函数
        model = model_fn()
        return model


if __name__ == '__main__':
    image_path = '/media/caoqi/HD2/Datasets/Dishcloth/JPEGImages/FF1227_YDXJ0514.jpg'
    checkpoint_dir = '/media/caoqi/HD2/Training/dishcloth3/training/best_model'
    pipeline_config_path = '/media/caoqi/HD2/Training/dishcloth3/dishcloth_supporter.config'
    label_map_path = '/media/caoqi/HD2/Training/dishcloth3/dishcloth_label_map.pbtxt'

    model = Model(checkpoint_dir, pipeline_config_path, label_map_path, (480, 640))

    video = cv2.VideoCapture(0)
    while True:
        ret, frame = video.read()

        drawn = model.visualize_prediction(frame, score_thres=0.6)
        cv2.imshow('window', drawn)
        cv2.waitKey(1)