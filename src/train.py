import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T

def get_transform():
    return T.Compose([
        T.ToTensor(),
        T.RandomHorizontalFlip(0.5)
    ])

def get_coco_dataset(root, annFile, transforms):
    dataset = CocoDetection(root=root, annFile=annFile, transform=transforms)
    return dataset

def get_model(num_classes):
    # Load a pre-trained model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    for images, targets in data_loader:
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

def evaluate(model, data_loader, device):
    model.eval()
    with torch.no_grad():
        for images, targets in data_loader:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            model(images, targets)

if __name__ == "__main__":
    train_dataset = get_coco_dataset('/path/to/coco/train2017', '/path/to/coco/annotations/instances_train2017.json', get_transform())
    val_dataset = get_coco_dataset('/path/to/coco/val2017', '/path/to/coco/annotations/instances_val2017.json', get_transform())

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)

    model = get_model(num_classes=91)  # COCO has 80 classes + 1 background

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    num_epochs = 10

    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=10)
        evaluate(model, val_loader, device)

    torch.save(model.state_dict(), 'models/fasterrcnn_resnet50_fpn.pth')
