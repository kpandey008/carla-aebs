import torchvision.transforms as T

from perception.trainer import Trainer
from perception.dataset import PerceptionDataset
from perception.model import PerceptionNet


base_dir = '/home/lexent/carla_simulation/perception_data/'
save_path = '/home/lexent/carla_simulation/chkpt/'
seed = 0

# Define the transforms
transform = T.Compose([
    T.ToTensor(),
])
train_dataset = PerceptionDataset(base_dir, seed=seed, mode='train', transform=transform)
val_dataset = PerceptionDataset(base_dir, seed=seed, mode='val', transform=transform)
model = PerceptionNet()

trainer = Trainer(train_dataset, val_dataset, model, random_state=seed)
trainer.train(save_path)
