import click
import yaml
import lightning as L
from lit_module import LitModule
from lightning.pytorch.loggers import TensorBoardLogger




@click.group()
def train():
    pass

@train.command()
@click.option("--config_path", required=True, help="Path to the config yaml.")
def new(config_path):
    hparams_dict = read_yaml_file_into_dict(config_path)
    lit_module = LitModule(**hparams_dict)
    start_training(lit_module)

@train.command()
@click.option("--checkpoint_path", required=True, help="Path to model checkpoint.")
def resume(checkpoint_path):
    lit_module = LitModule.load_from_checkpoint(checkpoint_path)
    start_training(lit_module,resume_from_checkpoint=checkpoint_path)

@train.command()
@click.option("--config_path", required=True, help="Path to the config yaml.")
@click.option("--checkpoint_path", required=True, help="Path to model checkpoint.")
def modify(config_path, checkpoint_path):
    hparams_dict = read_yaml_file_into_dict(config_path)
    lit_module = LitModule.load_from_checkpoint(checkpoint_path,strict=False,**hparams_dict)
    start_training(lit_module)
    

def read_yaml_file_into_dict(yaml_file_path):
    with open(yaml_file_path) as f:
        return yaml.safe_load(f)

def start_training(lit_module,resume_from_checkpoint=None):
    p = lit_module.hparams

    print_hparams(p)
    

    logger = TensorBoardLogger("lightning_logs", name="my_model")
    

    trainer = L.Trainer(
        accelerator=p.device, 
        max_epochs=p.max_epochs,
        devices=1,
        logger=logger,
        log_every_n_steps=100,
        )

    trainer.fit(
        model=lit_module,
        ckpt_path=resume_from_checkpoint,
    )

def print_hparams(p):
    print()
    print("Hyper Parameters:")
    for k,v in p.items():
        print(f"\t{k}: {v}")
    print()

if __name__ == "__main__":
    train()