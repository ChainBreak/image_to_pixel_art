import numpy as np
import torch
import lightning as L
import torchvision
import torch.nn.functional as F
from pathlib import Path
from models.encoder import Encoder
from models.decoder import Decoder
from models.quantizer import Quantizer
import time

class LitModule(L.LightningModule):
    def __init__(self, **kwargs):

        super(LitModule, self).__init__()

        self.save_hyperparameters()
        p = self.hparams

        self.encoder_model = Encoder(
            num_channels = [16, 32, 64, 128],
            num_blocks = [2, 2, 3, 4],
        )
        self.decoder_model = Decoder(
            num_channels = [16, 32, 64, 128],
            num_blocks = [2, 2, 3, 4],
        )

        self.quantizer = Quantizer(  channels=128, color_pallete=p.color_pallete  )
 
        self.example_input_array = torch.randn(1, 3, 256, 256)

        self.last_image_log_time = 0

    def forward(self, x):
        x = self.encoder_model(x)
        x = self.quantizer(x)
        x = self.decoder_model(x)

        return x

    def training_step(self, batch, batch_idx):
        input_image, y = batch

        x = self.encoder_model(input_image)

        pixel_art = self.quantizer(x)

        x_hat = self.decoder_model(pixel_art)

        x_low_res = F.interpolate(input_image, size=pixel_art.shape[2:], mode='bilinear', align_corners=False)

        loss_reconstruction = F.mse_loss(x_hat, input_image)
        loss_low_res = F.mse_loss(pixel_art, x_low_res)

        low_res_weighting = 0.2 #np.interp(self.global_step, [0, 3000], [1.0, 0.0])

        loss = loss_reconstruction + loss_low_res * low_res_weighting

        self.log('train_loss_reconstruction', loss_reconstruction)
        self.log('train_loss_low_res', loss_low_res)
        self.log('train_loss', loss)
        self.log('low_res_weighting', low_res_weighting)
        self.log_batch_as_image_grid(pixel_art, 'train_pixel_art')
        self.log_batch_as_image_grid(input_image, 'input_image')
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
        p=self.hparams
        optimizer = torch.optim.Adam(self.parameters(), lr=p.learning_rate)
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