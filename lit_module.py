import torch
from torch import nn
import lightning as L
import torchvision
import torch.nn.functional as F
from pathlib import Path
from models.encoder import Encoder
from models.decoder import Decoder

class LitModule(L.LightningModule):
    def __init__(self, **kwargs):

        super(LitModule, self).__init__()

        self.save_hyperparameters()

        self.encoder_model = Encoder(  down_blocks_channels = [16, 32, 64]  )
        self.decoder_model = Decoder(  down_blocks_channels = [16, 32, 64]  )
 
        self.example_input_array = torch.randn(1, 3, 128, 128)

    def forward(self, x):
        x = self.encoder_model(x)
        x = self.decoder_model(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        pixel_art = self.encoder_model(x)
        x_hat = self.decoder_model(pixel_art)

        loss = F.mse_loss(x_hat, x)
        self.log('train_loss', loss)
        print(f"train_loss: {loss}")
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
            torchvision.transforms.Resize((256 , 256)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5,))
        ])
        return transform(pil_image)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer