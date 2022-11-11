from random import shuffle
import torch
from torch.utils.data import Dataset, DataLoader
from faster_rcnn import create_model
from datasets import ElevatorButtonDataset
from prepare_data import valid_classes, get_train_transform, get_test_transforms, collate_fn, Averager
import matplotlib.pyplot as plt
import os

import detectron2
from detectron2 import model_zoo
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg

def get_EB_datadicts():
    # NOTE: the RESIZE H and W are not used in this training. For detectron training, image resize needs to be
    # added as an augmentation.
    RESIZE_H = 480
    RESIZE_W = 640
    CLASSES = valid_classes # currently have 38 classes
    BATCH_SIZE = 16
    DEVICE = 'cuda:0'

    data_root = 'your folder'
    train_dir = os.path.join(data_root, 'ElevatorButtonDataset/iros2018/train_set')
    test_dir = os.path.join(data_root, 'ElevatorButtonDataset/iros2018/test_set')

    train_dataset = ElevatorButtonDataset(train_dir, RESIZE_H,RESIZE_W, CLASSES, get_train_transform())
    valid_dataset = ElevatorButtonDataset(test_dir, RESIZE_H,RESIZE_W, CLASSES, get_test_transforms())

    return train_dataset.get_datadicts()

def demo_data():
    DatasetCatalog.register('EB_train', get_EB_datadicts)
    data = DatasetCatalog.get('EB_train')
    print('number of samples ', len(data))
    print('each looks like ', data[0])

def train():
    """
    TODO: add validation in DatasetCatalog; add image resizing to match our image size.
    """
    setup_logger()

    DatasetCatalog.register('EB_train', get_EB_datadicts)

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_C4_1x.yaml"))
    cfg.DATASETS.TRAIN = ("EB_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 8
    cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_50_C4_1x/137257644/model_final_721ade.pkl"  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 3000    # 3000 iterations reduce loss from 4.5 to 0.3, train cls accuracy to 0.95
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 4096   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 38

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()
    print('Training started...')
    print('Train results and history saved to ', cfg.OUTPUT_DIR)


if __name__=='__main__':
    train()