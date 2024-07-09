import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection

def get_transform():
    return T.Compose([
        T.ToTensor(),
        T.RandomHorizontalFlip(0.5)
    ])

def get_coco_dataset(root, annFile, transforms):
    dataset = CocoDetection(root=root, annFile=annFile, transform=transforms)
    return dataset

train_dataset = get_coco_dataset('/path/to/coco/train2017', '/path/to/coco/annotations/instances_train2017.json', get_transform())
val_dataset = get_coco_dataset('/path/to/coco/val2017', '/path/to/coco/annotations/instances_val2017.json', get_transform())

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)
