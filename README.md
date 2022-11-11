# Elevator Button Detection

This repo implements using Faster RCNN model to detect elevator buttons.

### Requirements

1.  Ubuntu == 18.04
2.  pytorch 1.8.1
3.  Python 3
4.  detectron2

### How-to

First, install detectron2 following the official installation instruction [here](https://github.com/facebookresearch/detectron2).
Secondly, make sure you have the Elevator Button Dataset downloaded and unzipped. Modify the *data_root* to point to your folder location.
Finally, run
```
python ./models/train_detectron.py
```

For now, only use code inside *models* folder.

To download the dataset, go [here](https://mycuhk-my.sharepoint.com/personal/1155067732_link_cuhk_edu_hk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2F1155067732%5Flink%5Fcuhk%5Fedu%5Fhk%2FDocuments%2FElevatorButtonDataset%2Ezip&parent=%2Fpersonal%2F1155067732%5Flink%5Fcuhk%5Fedu%5Fhk%2FDocuments&ga=1)