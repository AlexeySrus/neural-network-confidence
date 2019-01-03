import torch
import tqdm
from research.utils.losses import l2


class Model:
    def __init__(self, net, _device='cpu', callbacks_list=None):
        self.device = torch.device("cpu" if _device == 'cpu' else "cuda")
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
            epochs: epochs count
            loss_function: Loss function
            validation_loader: DataLoader
            verbose: print evaluate validation prediction
            init_start_epoch: start epochs number
        Returns:
        """
        for epoch in range(init_start_epoch, epochs + 1):
            self.model.train()

            batches_count = len(train_loader)
            avg_epoch_loss = 0

            with tqdm.tqdm(total=batches_count) as pbar:
                for i, (_x, _y_true) in enumerate(train_loader):
                    x = _x.to(self.device)
                    y_true = _y_true.to(self.device)
                    optimizer.zero_grad()
                    y_pred = self.model(x)
                    loss = loss_function(y_pred, y_true)
                    loss.backward()
                    optimizer.step()

                    pbar.postfix = \
                        'Epoch: {}/{}, Loss: {:.8f}'.format(epoch,
                                                            epochs,
                                                            loss.item()
                                                            )
                    avg_epoch_loss += loss.item() / batches_count

                    for cb in self.callbacks:
                        cb.per_batch({
                            'model': self,
                            'loss': loss.item(),
                            'n': (epoch - 1)*batches_count + i + 1,
                            'x': x,
                            'y_pred': y_pred,
                            'y_true:': y_true
                        })

                    pbar.update(1)

            test_loss = None

            if validation_loader is not None:
                test_loss = self.evaluate(
                    validation_loader, loss_function, verbose
                )
                self.model.train()

            for cb in self.callbacks:
                cb.per_epoch({
                    'model': self,
                    'loss': avg_epoch_loss,
                    'val loss': test_loss,
                    'n': epoch,
                    'optimize_state': optimizer.state_dict()
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

        with torch.no_grad():
            set_range = tqdm.tqdm(test_loader) if verbose else test_loader
            for _x, _y_true in set_range:
                x = _x.to(self.device)
                y_true = _y_true.to(self.device)
                y_pred = self.model(x)
                test_loss += loss_function(y_pred, y_true).item()

        test_loss /= len(test_loader)

        return test_loss

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