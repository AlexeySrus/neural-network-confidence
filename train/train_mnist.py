import torch
import argparse
import os
from model.mnist_architectures import MNISTNet
from model.model import Model, get_last_epoch_weights_path
import torch.nn.functional as F
from utils.callbacks import (SaveModelPerEpoch, VisPlot,
                                      SaveOptimizerPerEpoch)
from torch.utils.data import DataLoader
from utils.loaders import load_mnist, get_loaders
import yaml


def parse_args():
    parser = argparse.ArgumentParser(description='MNIST train script')
    parser.add_argument('--config', required=True, type=str,
                        help='Path to configuration yml file.')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 1)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    return parser.parse_args()


def main():
    args = parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = 'gpu' if use_cuda else 'cpu'

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    batch_size = config['train']['batch_size']
    n_jobs = config['train']['number_of_processes']

    model = Model(MNISTNet(), device)

    callbacks = []

    callbacks.append(SaveModelPerEpoch(
        os.path.join(
            os.path.dirname(__file__),
            config['train']['save']['model']
        ),
        config['train']['save']['every']
    ))

    callbacks.append(SaveOptimizerPerEpoch(
        os.path.join(
            os.path.dirname(__file__),
            config['train']['save']['model']
        ),
        config['train']['save']['every']
    ))

    if config['visualization']['use_visdom']:
        plots = VisPlot(
            'MNIST classification model',
            server=config['visualization']['visdom_server'],
            port=config['visualization']['visdom_port']
        )

        plots.register_scatterplot('train validation loss per_epoch', 'Epochs',
                                   'Loss',
                                   [
                                       'train binary cross entropy',
                                       'validation binary cross entropy'
                                   ])

        plots.register_scatterplot('train validation acc per_epoch', 'Epochs',
                                   'acc',
                                   [
                                       'train acc',
                                       'validation acc'
                                   ])

        callbacks.append(plots)

    model.set_callbacks(callbacks)

    start_epoch = 0
    optimizer = torch.optim.Adam(
        model.model.parameters(),
        lr=config['train']['lr'],
        weight_decay=1E-4,
        amsgrad=True
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        args.epochs // 50,
        gamma=0.5
    )

    if config['train']['load']:
        weight_path, optim_path, start_epoch = get_last_epoch_weights_path(
            os.path.join(
                os.path.dirname(__file__),
                config['train']['save']['model']
            ),
            print
        )

        if weight_path is not None:
            model.load(weight_path)
            optimizer.load_state_dict(torch.load(optim_path))

    train_loader, val_loader = get_loaders(load_mnist())

    train_dataset = DataLoader(
        train_loader, batch_size=batch_size, num_workers=n_jobs
    )

    val_dataset = DataLoader(
        val_loader, batch_size=batch_size, num_workers=n_jobs
    )

    model.fit(
        train_dataset,
        (optimizer, scheduler),
        args.epochs,
        F.binary_cross_entropy,
        init_start_epoch=start_epoch + 1,
        validation_loader=val_dataset
    )


if __name__ == '__main__':
    main()