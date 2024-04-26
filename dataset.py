import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from copy import deepcopy

class CoordinatesDataset(Dataset):
    def __init__(self, data_dir, coords_df, transform=None, transform_x=None, transform_y=None):
        self.data_dir = data_dir
        self.transform = transform
        self.coords_df = deepcopy(coords_df)
        print(self.coords_df.dtypes)

        if transform_x is not None:
            self.coords_df.x = transform_x(self.coords_df.x)
        if transform_y is not None:
            self.coords_df.y = transform_y(self.coords_df.y)

    def __len__(self):
        return len(self.coords_df)

    def __getitem__(self, idx):

        row = self.coords_df.iloc[idx]

        image_id = self.coords_df.iat[idx, 0]
        x = row.x
        y = row.y

        filename = os.path.join(self.data_dir, str(int(image_id)))+  ".jpg"
        image = cv2.imread(filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = torch.from_numpy(np.array([x, y], dtype=np.float32) )

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        return image, label

class Reader(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.files = os.listdir(self.data_dir)
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):


        filename = os.path.join(self.data_dir, self.files[idx])
        image = cv2.imread(filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


        if self.transform is not None:
            image = self.transform(image=image)["image"]

        return image
