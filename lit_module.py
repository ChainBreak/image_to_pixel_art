import torch
from torch import nn
import lightning as L
import torchvision
import torch.nn.functional as F
from pathlib import Path
from models.encoder import Encoder
from models.decoder import Decoder
import time

class LitModule(L.LightningModule):
    def __init__(self, **kwargs):

        super(LitModule, self).__init__()

        self.save_hyperparameters()

        self.encoder_model = Encoder(  down_blocks_channels = [16, 32, 64]  )
        self.decoder_model = Decoder(  down_blocks_channels = [16, 32, 64]  )
 
        self.example_input_array = torch.randn(1, 3, 128, 128)

        self.last_image_log_time = 0

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
        self.log_batch_as_image_grid(pixel_art, 'train_pixel_art')
        self.log_batch_as_image_grid(x, 'input_image')
        self.log_batch_as_image_grid(x_hat, 'output_image')
        return loss
    
    def train_dataloader(self):
        p = self.hparams
        current_folder = Path(__file__).resolve().parent
        data_root = current_folder / '.dataset_root'

        # Define your training dataset and dataloader here
        train_dataset = torchvision.datasets.LFWPeople(
            root=data_root,
            split="train", 
            image_set="original",
            download=True, 
            transform = self.transform,
        )
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=p.batch_size,
            shuffle=True,
            )
        
        return train_loader

    def transform(self, pil_image):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((256 , 256)),
            torchvision.transforms.ToTensor(),
            # torchvision.transforms.Normalize((0.5,), (0.5,)),
        ])
        return transform(pil_image)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
    
    def log_batch_as_image_grid(self, batch, name):

        batch = batch[:4]

        if self.should_we_log_this_step():
            grid = torchvision.utils.make_grid(batch, nrow=2)
            self.log_image(grid, name)

    def log_image(self, image, name):
        if self.should_we_log_this_step():
            self.logger.experiment.add_image(name, image, self.global_step)

    def should_we_log_this_step(self):
        p = self.hparams

        current_time = time.time()

        if current_time - self.last_image_log_time > p.image_log_interval_seconds:
            self.last_image_log_time = current_time
            self.image_log_global_step = self.global_step

        return self.global_step == self.image_log_global_step