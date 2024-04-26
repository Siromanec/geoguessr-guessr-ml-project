import argparse
import os

import albumentations as A
import lightning as L
import numpy as np
import pandas as pd
import torch.nn
from albumentations.pytorch import ToTensorV2
from playsound import playsound
from pytorch_lightning.cli import ReduceLROnPlateau
from torch import nn
from torch.utils.data import DataLoader

import model_architectures.model_factory as model_factory
from dataset import CoordinatesDataset

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LitModel(L.LightningModule):
    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("LitModel")
        parser.add_argument("--lr", type=float, default=5e-2, help="learning rate")
        parser.add_argument("--gamma", type=float, default=0.5, help="decay of learning rate")
        parser.add_argument("--decay_frequency", type=float, default=1, help="decay of learning rate")
        return parent_parser

    def __init__(self, model, lr, gamma, classes=2, **kwargs):
        super().__init__()
        self.save_hyperparameters("lr", "gamma", "classes")
        self.model = model
        self.criterion = nn.MSELoss().to(DEVICE)
        self.mse = nn.MSELoss().to(DEVICE)

    def get_loss(self, batch, batch_idx):
        x, y = batch
        # y = torch.fft.fftn(y, self.hparams.classes, dim=1)
        y_hat = torch.sigmoid(self.model(x))

        loss = torch.exp(self.criterion(y_hat, y)) - 1
        with torch.no_grad():
            mse = self.mse(y_hat, y)
        return loss, mse
    def training_step(self, batch, batch_idx):
        loss, mse = self.get_loss(batch, batch_idx)
        self.log_dict({
            'train_loss': loss.item(),
            'train_mse': mse.item(),
        }, prog_bar=True)
        return loss
    def validation_step(self, batch, batch_idx):
        loss, mse = self.get_loss(batch, batch_idx)
        self.log_dict({
            'val_loss': loss.item(),
            'val_mse': mse.item(),
        }, prog_bar=True)
    def test_step(self, batch, batch_idx):
        loss, mse = self.get_loss(batch, batch_idx)
        logger = self.logger.experiment
        logger.a
        self.log_dict({
            'test_loss': loss.item(),
            'test_mse': mse.item(),
        }, prog_bar=True)
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.hparams.gamma)
        # lr_scheduler_config = {
        #     "scheduler": lr_scheduler,
        #     "interval": "epoch",
        #     "frequency": self.hparams.decay_frequency,
        # }
        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": lr_scheduler_config
        # }
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer, ..., patience=1, factor=.4),
                "monitor": "train_mse",
                "interval": "epoch",

                "frequency": 1,
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            },
        }




class LitDataModule(L.LightningDataModule):
    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("LitDataModule")
        parser.add_argument("--batch_size", type=int, default=128)
        parser.add_argument("--data_path", type=str, default="./data/Europe/")
        parser.add_argument("--num_workers", type=int, default=6)
        return parent_parser

    def __init__(self, data_path, batch_size, num_workers=6, pin_memory=True, pin_memory_device="cuda", **kwargs):
        super().__init__()
        self.save_hyperparameters("batch_size")
        self.data_path = data_path
        print(data_path)
        print(batch_size)

        min_x, min_y, max_x, max_y, = -9.277194, 35.226374, 30.923247, 58.888745

        self.bbox = (min_x, min_y, max_x, max_y)

        def normalize(data, m, M):
            return (data - m) / (M - m)

        def denormalize(data, m, M):
            return (data) * (M - m) + m

        self.transform_x = lambda x: normalize(x, min_x, max_x)
        self.transform_y = lambda y: normalize(y, min_y, max_y)
        self.itransform_x = lambda x: denormalize(x, min_x, max_x)
        self.itransform_y = lambda y: denormalize(y, min_y, max_y)

        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.pin_memory_device = pin_memory_device
        self.persistent_workers = self.num_workers > 0

        ...

    # def prepare_data(self):
    #     ...
    def setup_fit(self):
        train_transform = A.Compose(
            [
                A.SmallestMaxSize(max_size=192),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.5),
                A.RandomCrop(height=128, width=128),
                # A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
                # A.RandomBrightnessContrast(p=0.5),
                A.CoarseDropout(max_width=15, max_height=15, p=0.5),
                A.Normalize(),
                ToTensorV2(),
            ]
        )
        val_transform = A.Compose(
            [
                A.SmallestMaxSize(max_size=192),
                A.CenterCrop(height=128, width=128),
                A.Normalize(),
                ToTensorV2(),
            ]
        )

        dtypes = {"id": np.uint64, "x": float, "y": float}
        train_coords_df = pd.read_csv(os.path.join(self.data_path, "train_coords.csv"), dtype=dtypes)
        val_coords_df = pd.read_csv(os.path.join(self.data_path, "val_coords.csv"), dtype=dtypes)
        self.train_dataset = CoordinatesDataset(self.data_path,
                                                train_coords_df,
                                                train_transform,
                                                self.transform_x,
                                                self.transform_y)
        self.val_dataset = CoordinatesDataset(self.data_path,
                                              val_coords_df,
                                              val_transform,
                                              self.transform_x,
                                              self.transform_y)

    def setup_test(self):
        test_transform = A.Compose(
            [
                A.SmallestMaxSize(max_size=192),
                A.CenterCrop(height=128, width=128),
                A.Normalize(),
                ToTensorV2(),
            ]
        )
        test_coords_df = pd.read_csv(os.path.join(self.data_path, "test_coords.csv"))
        self.test_dataset = CoordinatesDataset(self.data_path,
                                                test_coords_df,
                                                test_transform,
                                                self.transform_x,
                                                self.transform_y)

    def setup(self, stage: str):
        match stage:
            case "fit":
                self.setup_fit()
            case "test":
                self.setup_test()

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            pin_memory_device=self.pin_memory_device,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            pin_memory_device=self.pin_memory_device,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            pin_memory_device=self.pin_memory_device,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )


if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    torch.backends.cuda.matmul.allow_tf32 = True

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='./models/')
    parser.add_argument('--model_name', type=str, default='ResNet18')
    parser.add_argument('--max_epochs', type=int, default=10)
    temp_args, _ = parser.parse_known_args()

    parser = model_factory.configure_argparser(parser, temp_args.model_name)
    parser = LitDataModule.add_argparse_args(parser)
    parser = LitModel.add_argparse_args(parser)

    args = parser.parse_args()
    dict_args = vars(args)

    data_module = LitDataModule(**dict_args)

    model = model_factory.get_model(**dict_args)

    lit_model = LitModel(model, **dict_args).to(DEVICE)
    trainer = L.Trainer(max_epochs=args.max_epochs)

    # trainer.test
    try:
        trainer.fit(
            lit_model,
            datamodule=data_module
        )
    except Exception as e:
        print()
        print("ERROR : " + str(e))
    finally:
        model_path = f'models/model_{trainer.logger.version}.pth'

        print("Saving model")
        print(model_path)
        torch.save(model.state_dict(), model_path)
        print("Model saved")
        playsound('./sound/metal-pipe.mp3')

