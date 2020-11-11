import click
import torchvision.transforms as T

from perception.trainer import Trainer
from perception.dataset import PerceptionDataset
from perception.model import PerceptionNet


@click.argument(base_dir)
@click.argument(save_path)
@click.option('--random-state', default=0, type=int)
@click.command()
def train_perception(base_dir, save_path, random_state=0):
    # Define the transforms
    transform = T.Compose([
        T.ToTensor(),
    ])
    train_dataset = PerceptionDataset(base_dir, seed=seed, mode='train', transform=transform)
    val_dataset = PerceptionDataset(base_dir, seed=seed, mode='val', transform=transform)
    model = PerceptionNet()

    trainer = Trainer(train_dataset, val_dataset, model, random_state=seed)
    trainer.train(save_path)


if __name__ == '__main__':
    train_perception()
