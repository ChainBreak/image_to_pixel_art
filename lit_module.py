import torch
from torch import nn
import lightning as L
import torchvision
import torch.nn.functional as F
from pathlib import Path

class LitModule(L.LightningModule):
    def __init__(self, **kwargs):

        super(LitModule, self).__init__()

        self.save_hyperparameters()

        self.encoder_model = self.create_encoder()

    def create_encoder(self):

        return nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(32,32),
            nn.ReLU(),
            nn.Linear(32, 10),
        )


    def forward(self, x):
        x = self.encoder_model(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        return loss
    
    def train_dataloader(self):

        current_folder = Path(__file__).resolve().parent
        data_root = current_folder / '.dataset_root'

        # Define your training dataset and dataloader here
        train_dataset = torchvision.datasets.LFWPeople(
            root=data_root,
            split= 'train', 
            image_set="original",
            download=True, 
            transform = self.transform,
        )
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32)
        return train_loader

    def transform(self, pil_image):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((128 , 128)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5,))
        ])
        return transform(pil_image)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer