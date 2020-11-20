import click
import torchvision.transforms as T

from datasets.perception import PerceptionDataset
from models.perception.model import PerceptionNet
from utils.trainer import Trainer
from utils.util import get_loss


@click.argument('save-path')
@click.argument('base-dir')
@click.option('--random-state', default=0, type=int)
@click.option('--batch-size', default=32, type=int)
@click.option('--lr', default=0.01, type=float)
@click.option('--optim', default='Adam')
@click.option('--epochs', type=int, default=100)
@click.command()
def train_perception(base_dir, save_path, batch_size=32, random_state=0, lr=0.01, optim='Adam', epochs=100):
    """Trains the Perception network
    Sample command: python train_perception.py /kaggle/input/simulation/perception_data/ /kaggle/working/ \
                    --batch-size 64\
                    --random-state 0\
                    --lr 0.01\
                    --optim SGD\
                    --epochs 100
    Args:
        base_dir ([str]): Directory which stores the training images
        save_path ([str]): Path to store saved model checkpoints to
        batch_size (int, optional): Training batch size. Defaults to 32.
        random_state (int, optional): Random state to initialize params with. Defaults to 0.
        lr (float, optional): Training Learning Rate. Defaults to 0.01.
        optim (str, optional): Optimizer to use for training. Defaults to 'Adam'.
        epochs (int, optional): Number of epochs to train for. Defaults to 100.
    """
    # Define the transforms
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])
    train_dataset = PerceptionDataset(base_dir, seed=random_state, mode='train', transform=transform)
    val_dataset = PerceptionDataset(base_dir, seed=random_state, mode='val', transform=transform)
    model = PerceptionNet()
    train_loss = get_loss('mse')
    eval_loss = get_loss('mae')
    optimizer_kwargs = {
        'momentum': 0.9
    }

    trainer = Trainer(
        train_dataset, val_dataset, model, train_loss,
        eval_loss=eval_loss,
        random_state=random_state,
        batch_size=batch_size,
        lr=lr, optim=optim, num_epochs=epochs, optimizer_kwargs=optimizer_kwargs
    )
    trainer.train(save_path)


if __name__ == '__main__':
    train_perception()
