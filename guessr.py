
import argparse
import os

import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import Reader
from model_architectures import model_factory
from train_model import DEVICE, LitDataModule
import albumentations as A

def predict_directory(dir, model, transform):
    reader = Reader(dir, transform=transform)
    dataloader = DataLoader(
        reader,
        batch_size=32,
        shuffle=False,
        pin_memory_device=str(DEVICE),
        pin_memory=True,
        num_workers=min(os.cpu_count(), len(reader) // 64 + 1),
        persistent_workers=True,
    )

    model.eval()
    datamodule = LitDataModule(dir, 32)
    results = []
    with torch.no_grad():
        for images in tqdm(dataloader):
            images = images.to(DEVICE)
            # pred_labels = (torch.fft.irfftn(model(images), 128, dim=1)[:,:2]).cpu().numpy()
            pred_labels = (torch.sigmoid(model(images))).cpu().numpy()


            # for i in pred_labels:
            pred_labels[:,0] = datamodule.itransform_x(pred_labels[:,0])
            pred_labels[:,1] = datamodule.itransform_y(pred_labels[:,1])

            results.append(pred_labels)

    results = np.concatenate(results)
    pd.DataFrame(results, reader.files).to_csv('results.csv')

    ...
def predict_file(file, model, transform):
    model.eval()
    datamodule = LitDataModule(dir, 32)
    results = []
    with torch.no_grad():

        image = cv2.imread(file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


        image = transform(image=image)["image"]
        image = image.reshape((1, *image.shape))

        images = image.to(DEVICE)
        # pred_labels = (torch.fft.irfftn(model(images), 128, dim=1)[:,:2]).cpu().numpy()
        pred_labels = (torch.sigmoid(model(images))).cpu().numpy()


        # for i in pred_labels:
        pred_labels[:,0] = datamodule.itransform_x(pred_labels[:,0])
        pred_labels[:,1] = datamodule.itransform_y(pred_labels[:,1])

        results.append(pred_labels)

    results = np.concatenate(results)
    print(tuple(*results))
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="Path to the folder containing images or path to the image")

    args = parser.parse_args()

    if not os.path.exists(args.path):
        print("Path '{}' does not exist!".format(args.path))
        exit(1)

    model_dict = torch.load("models/model_73.pth")
    model = model_factory.get_model("ResNet18", classes=2)
    model.load_state_dict(model_dict)
    model = model.to(DEVICE)

    transform = A.Compose(
        [
            A.SmallestMaxSize(max_size=192),
            A.CenterCrop(height=128, width=128),
            A.Normalize(),
            ToTensorV2(),
        ]
    )

    if os.path.isdir(args.path):
        predict_directory(args.path, model,transform)
        exit(0)
    if os.path.isfile(args.path):
        predict_file(args.path, model, transform)
        exit(0)
    print("unrecognized path")
    exit(1)
