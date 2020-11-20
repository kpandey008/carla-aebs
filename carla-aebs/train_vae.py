# Python module to learn a VAE for the Perception net training
import click
import numpy as np
import torch
import torchvision
import torchvision.transforms as T

from datasets.perception import PerceptionDataset
from models.icad.vae import VAE
from utils.criterion import VAELoss
from utils.trainer import VAETrainer
from utils.util import get_loss

import matplotlib.pyplot as plt


@click.group()
def cli():
    pass


@click.option('--random-state', default=0, type=int)
@click.option('--code-size', default=1024, type=int)
@click.option('--batch-size', default=32, type=int)
@click.option('--optim', default='Adam')
@click.option('--lr', default=0.001, type=float)
@click.option('--epochs', type=int, default=100)
@click.argument('save-path')
@click.argument('base-dir')
@cli.command()
def train_vae(base_dir, save_path, batch_size=32, code_size=1024, random_state=0, lr=0.001, optim='Adam', epochs=100):
    # Define the transforms
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])
    train_dataset = PerceptionDataset(base_dir, seed=random_state, mode='train', transform=transform)
    val_dataset = PerceptionDataset(base_dir, seed=random_state, mode='val', transform=transform)
    train_loss = VAELoss()
    eval_loss = get_loss('mse')
    model = VAE(code_size=code_size)

    trainer = VAETrainer(train_dataset, val_dataset, model, train_loss,
        eval_loss=eval_loss, random_state=random_state,
        batch_size=batch_size, lr=lr, optimizer=optim,
        num_epochs=epochs
    )
    trainer.train(save_path)


@click.option('--code', type=int, default=512)
@click.option('--num-samples', type=int, default=1)
@click.argument('load-path')
@cli.command()
# TODO: Update this method to infer latent code from the state dict
def generate_samples(load_path, code=512, num_samples=1):
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])
    # train_dataset = PerceptionDataset('/home/lexent/carla_simulation/perception_data/', seed=0, mode='train', transform=transform)
    train_dataset = PerceptionDataset('/home/lexent/carla_simulation/perception_data/', seed=0, mode='val', transform=transform)
    sample = train_dataset[2][0].unsqueeze(0)
    # Load the model checkpoint
    state_dict = torch.load(load_path)
    model = VAE(code_size=code)
    model.load_state_dict(state_dict['model'])
    model.eval()
    loss = torch.nn.MSELoss()

    # Generate latent codes
    z = torch.randn((num_samples, code))
    with torch.no_grad():
        decoder_out = model.decode(z)
        _, sample_out, _, _ = model(sample)
        print(loss(sample_out, sample))
    img = torchvision.utils.make_grid(decoder_out, nrow=1, padding=10).numpy()
    print(sample.shape)
    # plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.imshow(sample_out.squeeze().permute((1, 2, 0)))
    plt.show()


if __name__ == '__main__':
    cli()
