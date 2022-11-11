from random import shuffle
import torch
from torch.utils.data import Dataset, DataLoader
from faster_rcnn import create_model
from datasets import ElevatorButtonDataset
from prepare_data import valid_classes, get_train_transform, get_test_transforms, collate_fn, Averager
import matplotlib.pyplot as plt
import os


RESIZE_H = 480
RESIZE_W = 640
CLASSES = valid_classes
BATCH_SIZE = 16
DEVICE = 'cuda:0'

data_root = 'your folder'
train_dir = os.path.join(data_root, 'ElevatorButtonDataset/iros2018/train_set')
test_dir = os.path.join(data_root, 'ElevatorButtonDataset/iros2018/test_set')

train_dataset = ElevatorButtonDataset(train_dir, RESIZE_H,RESIZE_W, CLASSES, get_train_transform())
valid_dataset = ElevatorButtonDataset(test_dir, RESIZE_H,RESIZE_W, CLASSES, get_test_transforms())

train_loader = DataLoader(
    train_dataset,
    batch_size = BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    collate_fn=collate_fn
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size = BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn
)

def train(model, optimizer, dataloader, averager, epoch):
    # print(len(train_dataset))
    # print(train_dataset[0])
    print(f'Training epoch {epoch}...')
    averager.reset()
    for i, sample in enumerate(dataloader):
        optimizer.zero_grad()
        images, targets = sample

        images = list(image.to(DEVICE) for image in images)
        targets= list({k:v.to(DEVICE) for k,v in t.items()} for t in targets)

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()

        averager.update(loss_value)

        losses.backward()
        optimizer.step()
        print(f'  iteration {i}, loss {loss_value}')
    print(f'  epoch avg loss {averager.value}')

    return averager.value

def validate(model, dataloader, averager, epoch):
    # print(len(train_dataset))
    # print(train_dataset[0])
    print(f'Validate epoch {epoch}...')
    averager.reset()
    for i, sample in enumerate(dataloader):
        images, targets = sample

        images = list(image.to(DEVICE) for image in images)
        targets= list({k:v.to(DEVICE) for k,v in t.items()} for t in targets)

        with torch.no_grad():
            loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        averager.update(loss_value)
        print(f'  iteration {i}, loss {loss_value}')
    print(f'  epoch avg loss {averager.value}')
    return averager.value

if __name__ == '__main__':
    model = create_model(len(CLASSES))
    model = model.to(DEVICE)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    train_loss_list = []
    val_loss_list = []
    train_rec = Averager()
    val_rec = Averager()

    for epoch in range(80):
        train_loss = train(model, optimizer, train_loader, train_rec, epoch)
        val_loss = validate(model, valid_loader, val_rec, epoch)

        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)

        if (epoch+1)%10==0:
            torch.save(model.state_dict(), f'saved/faster_rcnn_epoch{epoch+1}.pth')

    fig, ax = plt.subplots()
    ax.plot(train_loss_list, color='blue')
    ax.plot(val_loss_list, color='red')
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    fig.savefig('saved/losses.png')
    