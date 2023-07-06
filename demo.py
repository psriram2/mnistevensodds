import os
import numpy as np

import argparse
import lightning as L
import torch
from PIL import Image
from lightning.pytorch.loggers import CSVLogger
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from torchvision import transforms

from datasets.MNIST import MNIST
from models.simple_cnn import SimpleCNN


# for reproducibility
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)


# model adapted from https://lightning.ai/docs/pytorch/stable/notebooks/lightning_examples/mnist-hello-world.html
class LitMNIST(L.LightningModule):
    def __init__(self, data_dir, batch_size, num_classes, hidden_size=64, learning_rate=2e-4):
        super().__init__()
        self.save_hyperparameters()

        # Set our init args as class attributes
        self.data_dir = data_dir
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        # Hardcode some dataset specific attributes
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.dims = (1, 28, 28)
        channels, width, height = self.dims
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        # Define PyTorch model
        self.model = SimpleCNN(num_classes)

        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.val_accuracy.update(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_accuracy, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.test_accuracy.update(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_accuracy, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run inference evens/odds classifier on MNIST')
    parser.add_argument('--image_path', type=str, required=True, help='image path for inference')
    parser.add_argument('--checkpoint', type=str, required=True, help='checkpoint for testing')
    parser.add_argument('--data_dir', type=str, required=True, help='root directory of dataset')

    args = parser.parse_args()

    # initialize with dummy args 
    model = LitMNIST(args.data_dir, 1, 2)
    model = LitMNIST.load_from_checkpoint(args.checkpoint)

    im = np.asarray(Image.open(args.image_path), dtype=np.float32)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    result = model.model(transform(im / 255.).cuda())
    result = F.log_softmax(result, dim=1)

    # print("result: ", result)

    result = torch.argmax(result, dim=1)

    if result[0].item() == 0:
        print("Even number")
    else:
        print("Odd number")
    
    
            
