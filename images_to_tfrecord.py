# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Convert raw RSNA dataset to TFRecord for pneumonia detection.
Example usage:
    python images_to_tfrecord.py --logtostderr \
      --train_image_dir="${TRAIN_IMAGE_DIR}" \
      --val_image_dir="${VAL_IMAGE_DIR}" \
      --test_image_dir="${TEST_IMAGE_DIR}" \
      --train_annotations_file="${TRAIN_ANNOTATIONS_FILE}" \
      --val_annotations_file="${VAL_ANNOTATIONS_FILE}" \
      --testdev_annotations_file="${TESTDEV_ANNOTATIONS_FILE}" \
      --output_dir="${OUTPUT_DIR}"
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import hashlib
import io
import json
import multiprocessing
import os
from absl import app
from absl import flags
import numpy as np
import PIL.Image

import dataset_util
import label_map_util

import tensorflow as tf

flags.DEFINE_string('train_image_dir', './generated_images', 'Training image directory.')
flags.DEFINE_string('val_image_dir', './stage_1_validation_images', 'Validation image directory.')
flags.DEFINE_string('test_image_dir', './generated_images/shift_image', 'Test image directory.')
flags.DEFINE_string('train_object_annotations_file', './object_annotation.json', '')
flags.DEFINE_string('val_object_annotations_file', './validation_object_annotation.json', '')
flags.DEFINE_string('train_caption_annotations_file', './caption_annotation.json', '')
flags.DEFINE_string('val_caption_annotations_file', './validation_annotation.json', '')
flags.DEFINE_string('output_dir', '/tmp/', 'Output data directory.')

FLAGS = flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)


def create_tf_example(image,
                      bbox_annotations,
                      caption_annotations,
                      image_dir,
                      category_index):
    """Converts image and annotations to a tf.Example proto.
  Args:
    image: dict with keys:
      [u'license', u'file_name', u'coco_url', u'height', u'width',
      u'date_captured', u'flickr_url', u'id']
    bbox_annotations:
      list of dicts with keys:
      [u'segmentation', u'area', u'iscrowd', u'image_id',
      u'bbox', u'category_id', u'id']
      Notice that bounding box coordinates in the official COCO dataset are
      given as [x, y, width, height] tuples using absolute coordinates where
      x, y represent the top-left (0-indexed) corner.  This function converts
      to the format expected by the Tensorflow Object Detection API (which is
      which is [ymin, xmin, ymax, xmax] with coordinates normalized relative
      to image size).
    image_dir: directory containing the image files.
    category_index: a dict containing COCO category information keyed
      by the 'id' field of each category.  See the
      label_map_util.create_category_index function.
  Returns:
    example: The converted tf.Example
    num_annotations_skipped: Number of (invalid) annotations that were ignored.
  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid PNG
  """
    image_height = image['height']
    image_width = image['width']
    filename = image['file_name']
    image_id = image['id']

    full_path = os.path.join(image_dir, filename)
    with tf.gfile.GFile(full_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    key = hashlib.sha256(encoded_jpg).hexdigest()

    xmin = []
    xmax = []
    ymin = []
    ymax = []
    is_crowd = []
    category_names = []
    category_ids = []
    area = []
    encoded_mask_png = []
    num_annotations_skipped = 0
    for object_annotations in bbox_annotations:
        (x, y, width, height) = tuple(object_annotations['bbox'])
        if width <= 0 or height <= 0:
            num_annotations_skipped += 1
            continue
        if x + width > image_width or y + height > image_height:
            num_annotations_skipped += 1
            continue
        xmin.append(float(x) / image_width)
        xmax.append(float(x + width) / image_width)
        ymin.append(float(y) / image_height)
        ymax.append(float(y + height) / image_height)
        is_crowd.append(object_annotations['iscrowd'])
        category_id = int(object_annotations['category_id'])
        category_ids.append(category_id)
        category_names.append(category_index[category_id]['name'].encode('utf8'))
        area.append(object_annotations['area'])

    captions = []
    for caption_annotation in caption_annotations:
        captions.append(caption_annotation['caption'].encode('utf8'))

    feature_dict = {
        'image/height':
            dataset_util.int64_feature(image_height),
        'image/width':
            dataset_util.int64_feature(image_width),
        'image/filename':
            dataset_util.bytes_feature(filename.encode('utf8')),
        'image/source_id':
            dataset_util.bytes_feature(str(image_id).encode('utf8')),
        'image/key/sha256':
            dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded':
            dataset_util.bytes_feature(encoded_jpg),
        'image/caption':
            dataset_util.bytes_list_feature(captions),
        'image/format':
            dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin':
            dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax':
            dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin':
            dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax':
            dataset_util.float_list_feature(ymax),
        'image/object/class/text':
            dataset_util.bytes_list_feature(category_names),
        'image/object/class/label':
            dataset_util.int64_list_feature(category_ids),
        'image/object/is_crowd':
            dataset_util.int64_list_feature(is_crowd),
        'image/object/area':
            dataset_util.float_list_feature(area),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return key, example, num_annotations_skipped


def _pool_create_tf_example(args):
    return create_tf_example(*args)


# bounding box annotations
def _load_object_annotations(object_annotations_file):
    tf.logging.info('Building object index.')
    with tf.gfile.GFile(object_annotations_file, 'r') as fid:
        img_to_obj_annotation = json.load(fid)

    category_index = label_map_util.create_category_index()

    images = []
    for patientId in img_to_obj_annotation:
        if patientId.endswith("1"):
            directory = "shift_image"
        elif patientId.endswith("2"):
            directory = "shift_bbox"
        elif patientId.endswith("3"):
            directory = "scale_bbox"
        elif patientId.endswith("4"):
            directory = "scale_image"
        elif patientId.endswith("5"):
            directory = "scale_shift_bbox"
        elif patientId.endswith("6"):
            directory = "shift_image_shift_bbox"
        else:
            directory = "scale_image_scale_shift_bbox"
        images.append({'height': 1024, 'width': 1024, 'id': patientId, 'file_name': "{}/{}.png".format(directory, patientId)})

    return images, img_to_obj_annotation, category_index


def _load_caption_annotations(caption_annotations_file):
    tf.logging.info('Building caption index.')
    with tf.gfile.GFile(caption_annotations_file, 'r') as fid:
        img_to_caption_annotation = json.load(fid)

    return img_to_caption_annotation


def _create_tf_record_from_rsna_annotations(
        object_annotations_file,
        caption_annotations_file,
        image_dir, output_path, num_shards):
    """Loads RSNA annotation json/csv files and converts to tf.Record format.
  Args:
    object_annotations_file: JSON file containing bounding box annotations.
    caption_annotations_file: JSON file containing caption annotations.
    image_dir: Directory containing the image files.
    output_path: Path to output tf.Record file.
    num_shards: Number of output files to create.
  """

    tf.logging.info('writing to output path: %s', output_path)
    writers = [
        tf.python_io.TFRecordWriter(output_path + '-%05d-of-%05d.tfrecord' %
                                    (i, num_shards)) for i in range(num_shards)
    ]

    images, img_to_obj_annotation, category_index = (
        _load_object_annotations(object_annotations_file))
    img_to_caption_annotation = (
        _load_caption_annotations(caption_annotations_file))

    pool = multiprocessing.Pool()
    total_num_annotations_skipped = 0
    for idx, (_, tf_example, num_annotations_skipped) in enumerate(
            pool.imap(_pool_create_tf_example,
                      [(image,
                        img_to_obj_annotation[image['id']],
                        img_to_caption_annotation[image['id']],
                        image_dir,
                        category_index)
                       for image in images])):
        if idx % 100 == 0:
            tf.logging.info('On image %d of %d', idx, len(images))

        total_num_annotations_skipped += num_annotations_skipped
        writers[idx % num_shards].write(tf_example.SerializeToString())

    pool.close()
    pool.join()

    for writer in writers:
        writer.close()

    tf.logging.info('Finished writing, skipped %d annotations.',
                    total_num_annotations_skipped)


def main(_):
    assert FLAGS.train_image_dir, '`train_image_dir` missing.'
    assert FLAGS.val_image_dir, '`val_image_dir` missing.'
    assert FLAGS.test_image_dir, '`test_image_dir` missing.'

    if not tf.gfile.IsDirectory(FLAGS.output_dir):
        tf.gfile.MakeDirs(FLAGS.output_dir)
    train_output_path = os.path.join(FLAGS.output_dir, 'train')
    val_output_path = os.path.join(FLAGS.output_dir, 'val')

    _create_tf_record_from_rsna_annotations(
        FLAGS.train_object_annotations_file,
        FLAGS.train_caption_annotations_file,
        FLAGS.train_image_dir,
        train_output_path,
        num_shards=256)
    _create_tf_record_from_rsna_annotations(
        FLAGS.val_object_annotations_file,
        FLAGS.val_caption_annotations_file,
        FLAGS.val_image_dir,
        val_output_path,
        num_shards=32)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    app.run(main)
