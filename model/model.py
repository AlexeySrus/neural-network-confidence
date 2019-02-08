import torch
import tqdm
import os
import re
from utils.losses import l2
from utils.losses import acc as acc_f
from utils.tensor_procesing import flatten


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


class Model:
    def __init__(self, net, _device='cpu', callbacks_list=None):
        self.device = torch.device('cpu' if _device == 'cpu' else 'cuda')
        self.model = net.to(self.device)
        self.callbacks = [] if callbacks_list is None else callbacks_list

    def fit(self,
            train_loader,
            optimizer,
            epochs=1,
            loss_function=l2,
            validation_loader=None,
            verbose=False,
            init_start_epoch=1):
        """
        Model train method
        Args:
            train_loader: DataLoader
            optimizer: optimizer from torch.optim with initialized parameters
            or tuple of (optimizer, scheduler)
            epochs: epochs count
            loss_function: Loss function
            validation_loader: DataLoader
            verbose: print evaluate validation prediction
            init_start_epoch: start epochs number
        Returns:
        """
        scheduler = None
        if type(optimizer) is tuple:
            scheduler = optimizer[1]
            optimizer = optimizer[0]

        for epoch in range(init_start_epoch, epochs + 1):
            self.model.train()

            batches_count = len(train_loader)
            avg_epoch_loss = 0
            avg_epoch_acc = 0

            if scheduler is not None:
                scheduler.step(epoch)

            with tqdm.tqdm(total=batches_count) as pbar:
                for i, (_x, _y_true) in enumerate(train_loader):
                    x = _x.to(self.device)
                    y_true = _y_true.to(self.device)

                    optimizer.zero_grad()
                    y_pred = self.model(x)

                    loss = loss_function(y_pred, y_true)
                    loss.backward()
                    optimizer.step()

                    acc = acc_f(
                        flatten(y_pred),
                        flatten(y_true)
                    )

                    pbar.postfix = \
                        'Epoch: {}/{}, loss: {:.8f}, acc: {:.8f}, lr: {:.8f}'.format(
                            epoch,
                            epochs,
                            loss.item() / train_loader.batch_size,
                            acc,
                            get_lr(optimizer)
                        )
                    avg_epoch_loss += \
                        loss.item() / train_loader.batch_size / batches_count

                    avg_epoch_acc += acc.detach().numpy() / batches_count

                    for cb in self.callbacks:
                        cb.per_batch({
                            'model': self,
                            'loss': loss.item() / train_loader.batch_size,
                            'n': (epoch - 1)*batches_count + i + 1,
                            'x': x,
                            'y_pred': y_pred,
                            'y_true': y_true,
                            'acc': acc.detach().numpy()
                        })

                    pbar.update(1)

            test_loss = None
            test_acc = None

            if validation_loader is not None:
                test_loss, test_acc = self.evaluate(
                    validation_loader, loss_function, verbose
                )
                self.model.train()

            for cb in self.callbacks:
                cb.per_epoch({
                    'model': self,
                    'loss': avg_epoch_loss,
                    'val loss': test_loss,
                    'n': epoch,
                    'optimize_state': optimizer.state_dict(),
                    'acc': avg_epoch_acc,
                    'val acc': test_acc
                })

    def evaluate(self,
                 test_loader,
                 loss_function=l2,
                 verbose=False):
        """
        Test model
        Args:
            test_loader: DataLoader
            loss_function: loss function
            verbose: print progress

        Returns:

        """
        self.model.eval()

        test_loss = 0
        test_acc = 0

        with torch.no_grad():
            set_range = tqdm.tqdm(test_loader) if verbose else test_loader
            for _x, _y_true in set_range:
                x = _x.to(self.device)
                y_true = _y_true.to(self.device)
                y_pred = self.model(x)
                test_loss += loss_function(
                    y_pred, y_true
                ).item() / test_loader.batch_size / len(test_loader)
                test_acc += \
                    acc_f(y_pred, y_true).detach().numpy() / len(test_loader)

        return test_loss, test_acc

    def predict(self,
                predict_loader,
                verbose=False):
        """
        Predict
        Args:
            predict_loader: DataLoader
            verbose: print prediction progress

        Returns:

        """
        y_pred = []

        with torch.no_grad():
            set_range = tqdm.tqdm(predict_loader) if verbose else predict_loader
            for _x in set_range:
                x = _x.to(self.device)
                y_pred.append(self.model(x))

        return torch.cat(y_pred, dim=0)

    def set_callbacks(self, callbacks_list):
        self.callbacks = callbacks_list

    def save(self, path):
        torch.save(self.model.cpu().state_dict(), path)
        self.model = self.model.to(self.device)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        self.model = self.model.to(self.device)


def get_last_epoch_weights_path(checkpoints_dir, log=None):
    """
    Get last epochs weights from target folder
    Args:
        checkpoints_dir: target folder
        log: logging, default standard print
    Returns:
        (
            path to current weights file,
            path to current optimiser file,
            current epoch number
        )
    """
    if not os.path.isdir(checkpoints_dir):
        os.makedirs(checkpoints_dir)
        return None, None, 0

    weights_files_list = [
        matching_f.group()
        for matching_f in map(
            lambda x: re.match('model-\d+.trh', x),
            os.listdir(checkpoints_dir)
        ) if matching_f if not None
    ]

    if len(weights_files_list) == 0:
        return None, None, 0

    weights_files_list.sort(key=lambda x: -int(x.split('-')[1].split('.')[0]))

    if log is not None:
        log('LOAD MODEL PATH: {}'.format(
            os.path.join(checkpoints_dir, weights_files_list[0])
        ))

    n = int(
        weights_files_list[0].split('-')[1].split('.')[0]
    )

    return os.path.join(checkpoints_dir,
                        weights_files_list[0]
                        ), \
           os.path.join(checkpoints_dir, 'optimize_state-{}.trh'.format(n)), n