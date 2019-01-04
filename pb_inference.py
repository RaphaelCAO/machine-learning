import re
import numpy as np
import os
import tensorflow as tf

import cv2
from object_detection.utils import ops as utils_ops


class Model(object):
    def __init__(self, PATH_TO_PB, PATH_TO_LABELS):
        self.detection_graph = tf.Graph()

        # load the labels
        with open(PATH_TO_LABELS) as f:
            text = str(f.readlines())
            pattern = r"name:\s?\'(\w*)\'"
            self.label_names = re.findall(pattern, text)

        # initialize 10 common colors
        self.colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 255, 0),
                       (255, 0, 255), (0, 165, 255), (147, 20, 255), (35, 142, 107), (255, 112, 132)]
        # initialize the rest colors for the additional labels
        if len(self.label_names) > 10:
            additional_colors = [(int(np.random.random()*255), int(np.random.random()*255), int(np.random.random()*255))
                                 for i in range(len(self.label_names)-10)]
            self.colors.extend(additional_colors)

        # import the graph
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_PB, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            self.sess = tf.Session(graph=self.detection_graph)
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            self.tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    self.tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)

            self.image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

    def run_inference_for_single_image(self, image):
        """
        inference one image
        :param image: image path
        :return: the prediction result
        """
        if type(image) is str:
            image_np = cv2.imread(image)
        else:
            image_np = image

        with self.detection_graph.as_default():
            if 'detection_masks' in self.tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(self.tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(self.tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(self.tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image_np.shape[0], image_np.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                self.tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)

            # Run inference
            output_dict = self.sess.run(self.tensor_dict, feed_dict={self.image_tensor: np.expand_dims(image_np, 0)})
            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
        return output_dict

    def visualize_prediction(self, image, score_thres=0.3, window_name='window'):
        """
        Given a picture path, visualize the bounding boxes of predicted objects.
        :param img: the path of an image
        :param score_thres: the score threshold for visible bounding box
        :return: None
        """
        output_dict = self.run_inference_for_single_image(image)
        image_np = cv2.imread(image) if type(image) is str else image

        scores = output_dict['detection_scores']
        boxes = output_dict['detection_boxes']
        labels = output_dict['detection_classes']

        height, width, _ = image_np.shape
        for score, box, label in zip(scores, boxes, labels):
            if score < score_thres:
                break
            score = int(score * 100) / 100
            pt1 = (int(box[1] * width), int(box[0] * height))
            pt2 = (int(box[3] * width), int(box[2] * height))
            color = self.colors[label-1]
            cv2.rectangle(image_np, pt1, pt2, color, 2)

            title = "{}: {}".format(self.label_names[label-1], score)
            bar_width = np.ceil(score * (pt2[0]-pt1[0])).astype(np.int)
            cv2.rectangle(image_np, (pt1[0], pt1[1]-35), (pt2[0], pt1[1]), color, 2)
            cv2.rectangle(image_np, (pt1[0], pt1[1]-35), (pt1[0]+bar_width, pt1[1]), color, cv2.FILLED)
            cv2.putText(image_np, title,
                        (pt1[0], pt1[1]-10), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)

        cv2.namedWindow(window_name, 0)
        cv2.imshow(window_name, image_np)

        return image_np


if __name__ == '__main__':
    PATH_TO_PB = '/home/raphael/pb_models/coco_masked.pb'
    PATH_TO_LABELS = '/home/raphael/pb_models/mscoco_label_map.pbtxt'

    PATH_TO_PB2 = '/home/raphael/pb_models/coco.pb'

    model1 = Model(PATH_TO_PB, PATH_TO_LABELS)
    model2 = Model(PATH_TO_PB2, PATH_TO_LABELS)

    im_path = ['/home/raphael/test.png',
               '/home/raphael/caoqi_0_ver1.jpg',
               '/home/raphael/caoqi_0_ver2.jpg']
    predicted_img1 = model1.visualize_prediction(im_path[2], score_thres=0.3, window_name='window1')
    predicted_img2 = model2.visualize_prediction(im_path[2], score_thres=0.3, window_name='window2')

    key = cv2.waitKey()
    if (key & 0xFF) == ord('q'):
        cv2.destroyAllWindows()
    elif (key & 0xFF) == ord('s'):
        cv2.imwrite('masked.jpg', predicted_img1)
        cv2.imwrite('unmasked.jpg', predicted_img2)


