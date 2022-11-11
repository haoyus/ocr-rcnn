from itertools import count
import os
import re
import random
import pickle
import shutil
import numpy as np
from lxml import etree
import matplotlib.pyplot as plt

import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

def recursive_parse_xml_to_dict(xml):
  """Recursively parses XML contents to python dict.

  We assume that `object` tags are the only ones that can appear
  multiple times at the same level of a tree.

  Args:
    xml: xml tree obtained by parsing XML file contents using lxml.etree

  Returns:
    Python dictionary holding XML contents.
  """
  if not xml:
    return {xml.tag: xml.text}
  result = {}
  for child in xml:
    child_result = recursive_parse_xml_to_dict(child)
    if child.tag != 'object':
      result[child.tag] = child_result[child.tag]
    else:
      if child.tag not in result:
        result[child.tag] = []
      result[child.tag].append(child_result[child.tag])
  return {xml.tag: result}

def get_image_name_list(target_path):
    if target_path is None:
        raise IOError('Target path cannot be found!')
    image_name_list = []
    file_set = os.walk(target_path)
    for root, dirs, files in file_set:
        for image_name in files:
            image_name_list.append(image_name.split('.')[0])
    return image_name_list

def prepare_dataset(name='all'):
    data_dir = 'path-to-data/ElevatorButtonDataset/iros2018/train_set'
    label_dir = os.path.join(data_dir, 'annotations')
    image_dir = os.path.join(data_dir, 'images')
    grids_dir = os.path.join(data_dir, 'grids')
    eval_dir = os.path.join(data_dir, 'evaluation_train+test')

    # list train and test samples
    sample_list = get_image_name_list(label_dir)
    random.seed(9420)
    random.shuffle(sample_list)
    num_samples = len(sample_list)
    num_train = int(0.8 * num_samples)
    train_samples = sample_list[:num_train]
    test_samples = sample_list[num_train:]

    if name == 'train':
        dataset_type = train_samples
    elif name == 'test':
        dataset_type = test_samples
    else:
        dataset_type = sample_list

    # process single sample to get adjacent matrix
    dataset = []
    for idx, example in enumerate(dataset_type):
        annotation_path = os.path.join(label_dir, example + '.xml')
        with open(annotation_path, 'r') as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = recursive_parse_xml_to_dict(xml)['annotation']

        data['folder'] = data_dir
        data['path'] = os.path.join(image_dir, data['filename'])
        for obj in data['object']:
          obj['name'] = supress_cls(obj['name'])

        dataset.append(data)
    return dataset, data_dir

def count_categories(dataset):
    cls_count = {}
    for sample in dataset:
        buttons = sample['object']
        for b in buttons:
            b_name = b['name']
            if b_name not in cls_count:
                cls_count[b_name] = 0
            cls_count[b_name] += 1

    sorted_count = sorted(cls_count.items(), key=lambda item:item[1], reverse=True)
    return cls_count, sorted_count

def get_image_path(root, filename):
  return os.path.join(root, filename)

valid_classes = [str(i) for i in range(30)]
valid_classes.extend(['G', 'B', 'B1', 'open', 'close', 'up', 'down', 'unknown'])
valid_classes_set = set(valid_classes)
def supress_cls(cls):
  if cls in valid_classes:
    return cls
  else:
    return 'unknown'


class Averager:
	def __init__(self) -> None:
		self.total = 0.
		self.iterations = 0

	def update(self, value):
		self.total += value
		self.iterations += 1

	@property
	def value(self):
		if self.iterations==0:
			return 0
		else:
			return self.total / float(self.iterations)

	def reset(self):
		self.total = 0.
		self.iterations = 0


def collate_fn(batch):
	return tuple(zip(*batch))

def image2annotation(image_path):
  anno = image_path.replace('/images/', '/annotations/')
  anno = anno[:-4] + '.xml'
  return anno

def get_train_transform():
	return A.Compose([
		A.MotionBlur(p=0.2),
		A.MedianBlur(blur_limit=3, p=0.1),
		ToTensorV2(p=1.)
	], bbox_params={
		'format': 'pascal_voc',
		'label_fields': ['labels']
	})

def get_test_transforms():
	return A.Compose([
		ToTensorV2(p=1.0)
	], bbox_params={
		'format': 'pascal_voc',
		'label_fields': ['labels']
	})

if __name__ == '__main__':
    dataset, _ = prepare_dataset()
    print(len(dataset))
    print(dataset[0])
    cls_count, sorted_count = count_categories(dataset)
    print(len(cls_count))
    print(sorted_count[:40])
    print(valid_classes)