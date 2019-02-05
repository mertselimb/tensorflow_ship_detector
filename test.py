

#  Mert Selimbeyoğlu 
#  05.02.2019 
#  mertselimb@gmail.com


import numpy as np
import os
import six.moves.urllib as urllib
import sys
import random
import tarfile
import tensorflow as tf
import zipfile
import argparse
from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from os import listdir
from os.path import isfile, join
import datetime


# argparser setup

parser = argparse.ArgumentParser()
parser.add_argument('-t','--test-amount', help='How many images should be used', required=False)
parser.add_argument('-s','--image-size', help='Plot image size by inch', required=False)
args = parser.parse_args()
# Size, in inches, of the output images.
if(args.image_size != None):
	IMAGE_SIZE = (int(float(args.image_size)) , int(float(args.image_size)))
else : 
	IMAGE_SIZE = (40, 40)
print("\033[92m Image size = " + str(IMAGE_SIZE))
print("\033[95m")

if(args.test_amount != None):
	TEST_AMOUNT = int(float(args.test_amount))
else : 
	TEST_AMOUNT = 10
print("\033[92m Test amount = " + str(TEST_AMOUNT))
print("\033[95m")

sys.path.append("..")
from object_detection.utils import ops as utils_ops

if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
  raise ImportError('\033[92m Please upgrade your TensorFlow installation to v1.9.* or later!')


# ## Object detection imports
# Here are the imports from the object detection module.

from utils import label_map_util

from utils import visualization_utils as vis_util


# # Model preparation 

MODEL_NAME = 'ship_detector_graph'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('training', 'object-detection.pbtxt')


# ## Load a (frozen) Tensorflow model into memory.

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `1`, we know that this corresponds to `ship`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)


# ## Helper code

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


# # Detection


PATH_TO_TEST_IMAGES_DIR = 'images/test'

images = [f for f in listdir(PATH_TO_TEST_IMAGES_DIR) if isfile(join(PATH_TO_TEST_IMAGES_DIR, f))]

selected = random.sample(images, TEST_AMOUNT)
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, file) for file in selected ]

def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})
      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict

now = datetime.datetime.now()
SAVE_FOLDER = now.strftime("%Y-%m-%d_%H:%M:%S")
if(os.path.exists(test_output)):
	os.mkdir( "test_output/" + SAVE_FOLDER );
else:
	os.mkdir( "test_output");
	os.mkdir( "test_output/" + SAVE_FOLDER );

for i,image_path in enumerate(TEST_IMAGE_PATHS):
    image = Image.open(image_path)
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = load_image_into_numpy_array(image)
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection.
    output_dict = run_inference_for_single_image(image_np, detection_graph)
    # Visualization of the results of a detection.
    print("\033[92m Detecting image " + str(i+1) + " : " + image_path)
    print("\033[95m")
    pred = output_dict['detection_scores'] >= 0.50
    show = False
    count = 0
    for i in range(len(output_dict['detection_scores'])):
        if(pred[i]!=False):
            show = True
            count += 1
            print("\033[92m class: ship, prediction: %s"% (output_dict['detection_scores'][i]))
            print("\033[95m")

    vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks'),
      use_normalized_coordinates=True,
      line_thickness=1)
     
    plt.figure(figsize=IMAGE_SIZE)
    plt.title(image_path + " detected : " + str(count),color="black",fontsize=25)
    plt.axis('off')
    plt.imshow(image_np)
    plt.savefig( "test_output/" + SAVE_FOLDER + "/" + image_path.replace('images/test/', '') + "_detected_" + str(count) + '.png' , bbox_inches = 'tight')

