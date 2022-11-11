import torch
import cv2
import numpy as np
import os
import glob as glob
from xml.etree import ElementTree as et

from torch.utils.data import Dataset, DataLoader
from prepare_data import collate_fn, get_train_transform, get_test_transforms, supress_cls, valid_classes, image2annotation

from detectron2.structures import BoxMode

class ElevatorButtonDataset(Dataset):
    def __init__(self, data_dir, height, width, classes, transforms=None):
        super().__init__()
        self.transforms = transforms
        self.data_dir = data_dir
        self.height = height
        self.width = width
        self.classes = classes

        self.image_paths = glob.glob(f'{self.data_dir}/images/*.jpg')

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image_width = image.shape[1]
        image_height = image.shape[0]
        image_resized = cv2.resize(image, (self.width, self.height))
        image_resized /= 255.0

        anno_path = image2annotation(img_path)

        boxes = []
        labels = []
        tree = et.parse(anno_path)
        root = tree.getroot()
        for member in root.findall('object'):
            # map the current object name to `classes` list to get...
            # ... the label index and append to `labels` list
            cls_label = supress_cls(member.find('name').text)
            labels.append(valid_classes.index(cls_label))

            # xmin = left corner x-coordinates
            xmin = int(member.find('bndbox').find('xmin').text)
            # xmax = right corner x-coordinates
            xmax = int(member.find('bndbox').find('xmax').text)
            # ymin = left corner y-coordinates
            ymin = int(member.find('bndbox').find('ymin').text)
            # ymax = right corner y-coordinates
            ymax = int(member.find('bndbox').find('ymax').text)

            # resize the bounding boxes according to the...
            # ... desired `width`, `height`
            xmin_final = (xmin/image_width)*self.width
            xmax_final = (xmax/image_width)*self.width
            ymin_final = (ymin/image_height)*self.height
            yamx_final = (ymax/image_height)*self.height
            boxes.append([xmin_final, ymin_final, xmax_final, yamx_final])

        # bounding box to tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # area of the bounding boxes
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # no crowd instances
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        # labels to tensor
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # prepare the final `target` dictionary
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        image_id = torch.tensor([idx])
        target["image_id"] = image_id

        if self.transforms:
            sample = self.transforms(
                image = image_resized,
                bboxes = target['boxes'],
                labels = target['labels']
            )
            image_resized = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes'])
            # target['labels']= torch.Tensor(sample['labels'], dtype=torch.int64)

        # target['image_path'] = img_path

        return image_resized, target

    def __len__(self):
        return len(self.image_paths)

    def get_datadicts(self):
        ret = []
        N = len(self.image_paths)
        for i in range(N):
            annotations, image_height, image_width = self.get_detectron_annotations(idx=i)
            d = dict(
                file_name=self.image_paths[i],
                image_id=i,
                height=image_height,
                width=image_width,
                annotations=annotations
            )
            ret.append(d)
        return ret
        
    def get_detectron_annotations(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        image_width = image.shape[1]
        image_height = image.shape[0]

        anno_path = image2annotation(img_path)

        annotations=[]
        tree = et.parse(anno_path)
        root = tree.getroot()
        for member in root.findall('object'):
            # map the current object name to `classes` list to get...
            # ... the label index and append to `labels` list
            cls_label = supress_cls(member.find('name').text)

            # xmin = left corner x-coordinates
            xmin = int(member.find('bndbox').find('xmin').text)
            # xmax = right corner x-coordinates
            xmax = int(member.find('bndbox').find('xmax').text)
            # ymin = left corner y-coordinates
            ymin = int(member.find('bndbox').find('ymin').text)
            # ymax = right corner y-coordinates
            ymax = int(member.find('bndbox').find('ymax').text)

            bbox = [xmin, ymin, xmax, ymax]
            annotations.append(dict(
                bbox=bbox,
                bbox_mode=BoxMode.XYXY_ABS,
                category_id=valid_classes.index(cls_label),
                iscrowd=0
            ))
        
        return annotations, image_height, image_width


def test_xml(annot_file_path):
    boxes = []
    labels = []
    tree = et.parse(annot_file_path)
    root = tree.getroot()
    for member in root.findall('object'):
        # map the current object name to `classes` list to get...
        # ... the label index and append to `labels` list
        labels.append(valid_classes.index(member.find('name').text))
            
        # xmin = left corner x-coordinates
        xmin = int(member.find('bndbox').find('xmin').text)
        # xmax = right corner x-coordinates
        xmax = int(member.find('bndbox').find('xmax').text)
        # ymin = left corner y-coordinates
        ymin = int(member.find('bndbox').find('ymin').text)
        # ymax = right corner y-coordinates
        ymax = int(member.find('bndbox').find('ymax').text)
            
        # resize the bounding boxes according to the...
        # ... desired `width`, `height`
        # xmin_final = (xmin/image_width)*self.width
        # xmax_final = (xmax/image_width)*self.width
        # ymin_final = (ymin/image_height)*self.height
        # yamx_final = (ymax/image_height)*self.height
        xmin_final = xmin #/image_width)*self.width
        xmax_final = xmax #/image_width)*self.width
        ymin_final = ymin #/image_height)*self.height
        yamx_final = ymax #/image_height)*self.height
            
        boxes.append([xmin_final, ymin_final, xmax_final, yamx_final])

    print(boxes)
    print(labels)

if __name__ == '__main__':
    test_xml('/media/haoyusun/DATA/ElevatorButtonDataset/iros2018/train_set/annotations/792.xml')