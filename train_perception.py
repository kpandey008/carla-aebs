import click
import torchvision.transforms as T

from perception.trainer import Trainer
from perception.dataset import PerceptionDataset
from perception.model import PerceptionNet


@click.argument('save-path')
@click.argument('base-dir')
@click.option('--random-state', default=0, type=int)
@click.option('--batch-size', default=32, type=int)
@click.option('--lr', default=0.01, type=float)
@click.command()
def train_perception(base_dir, save_path, batch_size=32, random_state=0, lr=0.01):
    # Define the transforms
    transform = T.Compose([
        T.ToTensor(),
    ])
    train_dataset = PerceptionDataset(base_dir, seed=random_state, mode='train', transform=transform)
    val_dataset = PerceptionDataset(base_dir, seed=random_state, mode='val', transform=transform)
    model = PerceptionNet()

    trainer = Trainer(train_dataset, val_dataset, model, random_state=random_state, batch_size=batch_size, lr=lr)
    trainer.train(save_path)


if __name__ == '__main__':
    train_perception()
