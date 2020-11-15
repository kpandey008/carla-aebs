import torch
import torchvision.transforms as T
import os
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
from sklearn.model_selection import train_test_split


class PerceptionDataset(Dataset):
    def __init__(self, base_dir, mode='train', transform=None, seed=0, **kwargs):
        assert mode in ['train', 'val']
        if not os.path.isdir(base_dir):
            raise Exception(f'The directory {base_dir} does not exist.')
        self.base_dir = base_dir
        self.mode = mode
        self.images = []
        self.target = []
        self.transform = transform

        for dir_ in tqdm(os.listdir(self.base_dir)):
            if not dir_.startswith('episode'):
                continue
            dir_path = os.path.join(self.base_dir, dir_)
            episode_target = np.load(os.path.join(dir_path, 'target.npy'))
            for f in sorted(os.listdir(dir_path)):
                if f.startswith('target'):
                    self.target.append(episode_target)
                    continue
                self.images.append(os.path.join(dir_path, f))
        self.target = np.concatenate(self.target, axis=0)

        # Perform the train-test split
        X_train, X_val, Y_train, Y_val = train_test_split(self.images, self.target, test_size=0.1, random_state=seed)
        if self.mode == 'train':
            self.images = X_train
            self.target = Y_train
        elif self.mode == 'val':
            self.images = X_val
            self.target = Y_val
        self.target = torch.tensor(self.target).float()

    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        target = self.target[idx]
        if self.transform is not None:
            image = self.transform(image)
        return image, target/120.0

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    base_dir = '/home/lexent/carla_simulation/perception_data/'
    mode = 'train'
    seed = 0
    dataset = PerceptionDataset(base_dir, seed=seed, mode=mode)
    print(len(dataset))
    fig = plt.figure()
    plt.imshow(dataset[500][0])
    plt.show()
