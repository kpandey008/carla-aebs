import copy
import gc
import os
import torch

from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.util import configure_device, get_loss, get_lr_scheduler, get_optimizer


class Trainer:
    def __init__(self, train_dataset, val_dataset, model, train_loss, lr_scheduler='poly',
        num_epochs=100, batch_size=32, lr=0.01, eval_loss=None,
        log_step=10, optimizer='SGD', backend='gpu', random_state=0, optimizer_kwargs={},
        lr_scheduler_kwargs={}, **kwargs
    ):
        # Create the dataset
        self.lr = lr
        self.random_state = random_state
        self.device = configure_device(backend)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.num_epochs = num_epochs
        self.log_step = log_step
        self.loss_profile = []
        self.batch_size = batch_size

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=0)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, drop_last=False, num_workers=0)
        self.model = model.to(self.device)
        # The parameter train_loss must be a callable
        self.train_criterion = train_loss
        # The parameter eval_loss must be a callable
        self.val_criterion = eval_loss
        self.optimizer = get_optimizer(optimizer, self.model, self.lr, **optimizer_kwargs)
        self.lr_scheduler = get_lr_scheduler(self.optimizer, self.num_epochs, sched_type=lr_scheduler, **lr_scheduler_kwargs)

        # Some initialization code
        torch.set_default_tensor_type('torch.FloatTensor')
        if self.device == 'gpu':
            # Set a deterministic CuDNN backend
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def train(self, save_path, restore_path=None):
        start_epoch = 0
        if restore_path is not None:
            # Load the model
            self.load(restore_path)

        best_eval = 0.0
        for epoch_idx in range(start_epoch, self.num_epochs):
            print(f'Training for epoch: {epoch_idx}')
            avg_epoch_loss = self.train_one_epoch()

            # Build loss profile
            self.loss_profile.append(avg_epoch_loss)

            # Evaluate the model
            if self.val_criterion is not None:
                val_eval = self.eval()
                print(f'Avg Loss for epoch: {avg_epoch_loss} Eval Loss: {val_eval}')
                if epoch_idx == 0:
                    best_eval = val_eval
                    self.save(save_path, epoch_idx, prefix='best')
                else:
                    if best_eval > val_eval:
                        # Save this model checkpoint
                        self.save(save_path, epoch_idx, prefix='best')
                        best_eval = val_eval
            else:
                print(f'Avg Loss for epoch:{avg_epoch_loss}')
                if epoch_idx % 10 == 0:
                    # Save the model every 10 epochs anyways
                    self.save(save_path, epoch_idx)

    def eval(self):
        self.model.eval()
        eval_loss = 0
        with torch.no_grad():
            for idx, (img_batch, target_batch) in enumerate(self.val_loader):
                self.optimizer.zero_grad()
                img_batch = img_batch.to(self.device)
                target_batch = target_batch.to(self.device)
                predictions = self.model(img_batch).squeeze()
                loss = self.val_criterion(predictions, target_batch)
                eval_loss += loss.item()
        return eval_loss / len(self.val_loader)

    def train_one_epoch(self):
        self.model.train()
        epoch_loss = 0
        tk0 = tqdm(self.train_loader)
        for idx, (img_batch, target_batch) in enumerate(tk0):
            self.optimizer.zero_grad()
            img_batch = img_batch.to(self.device)
            target_batch = target_batch.to(self.device)
            predictions = self.model(img_batch).squeeze()
            loss = self.train_criterion(predictions, target_batch)
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
            if idx % self.log_step == 0:
                tk0.set_postfix_str(f'Loss at step {idx + 1}: {loss.item()}')
        return epoch_loss / len(self.train_loader)

    def save(self, path, epoch_id, prefix=''):
        checkpoint_name = f'chkpt_{epoch_id}'
        path = os.path.join(path, prefix)
        checkpoint_path = os.path.join(path, f'{checkpoint_name}.pt')
        state_dict = {}
        model_state = copy.deepcopy(self.model.state_dict())
        model_state = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in model_state.items()}
        optim_state = copy.deepcopy(self.optimizer.state_dict())
        for state in optim_state['state'].values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cpu()

        state_dict['model'] = model_state
        state_dict['optimizer'] = optim_state
        state_dict['scheduler'] = self.lr_scheduler.state_dict()
        state_dict['epoch'] = epoch_id + 1
        state_dict['loss_profile'] = self.loss_profile

        os.makedirs(path, exist_ok=True)
        for f in os.listdir(path):
            if f.endswith('.pt'):
                os.remove(os.path.join(path, f))
        torch.save(state_dict, checkpoint_path)
        del model_state, optim_state
        gc.collect()

    def load(self, load_path):
        state_dict = torch.load(load_path)
        iter_val = state_dict.get('epoch', 0)
        self.loss_profile = state_dict.get('loss_profile', [])
        if 'model' in state_dict:
            print('Restoring Model state')
            self.model.load_state_dict(state_dict['model'])

        if 'optimizer' in state_dict:
            print('Restoring Optimizer state')
            self.optimizer.load_state_dict(state_dict['optimizer'])
            # manually move the optimizer state vectors to device
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)

        if 'scheduler' in state_dict:
            print('Restoring Learning Rate scheduler state')
            self.lr_scheduler.load_state_dict(state_dict['scheduler'])


class VAETrainer(Trainer):
    def train_one_epoch(self):
        self.model.train()
        epoch_loss = 0
        tk0 = tqdm(self.train_loader)
        for idx, (img_batch, _) in enumerate(tk0):
            self.optimizer.zero_grad()
            img_batch = img_batch.to(self.device)
            _, predictions, mu, logvar = self.model(img_batch)
            loss = self.train_criterion(img_batch, predictions, mu, logvar)
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
            if idx % self.log_step == 0:
                tk0.set_postfix_str(f'Loss at step {idx + 1}: {loss.item()}')
        return epoch_loss/ len(self.train_loader)

    def eval(self):
        self.model.eval()
        eval_loss = 0
        with torch.no_grad():
            for idx, (img_batch, _) in enumerate(self.val_loader):
                self.optimizer.zero_grad()
                img_batch = img_batch.to(self.device)
                _, predictions, mu, logvar = self.model(img_batch)
                loss = self.val_criterion(img_batch, predictions)
                eval_loss += loss.item()
        return eval_loss / len(self.val_loader)
