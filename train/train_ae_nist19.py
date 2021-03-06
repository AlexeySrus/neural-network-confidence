import torch
import argparse
import os
from model.nist19_architectures import NIST19Net2, ConfidenceAE2
from model.model import Model, get_last_epoch_weights_path
import torch.nn.functional as F
from utils.callbacks import (SaveModelPerEpoch, VisPlot,
                                      SaveOptimizerPerEpoch,
                                        VisImageForAE)
from torch.utils.data import DataLoader
from utils.loaders import NIST19Loader
import yaml


def parse_args():
    parser = argparse.ArgumentParser(description='NIST19 AE train script')
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

    train_loader = NIST19Loader(config['train']['data'], for_ae=True)
    val_loader = NIST19Loader(config['train']['data'], validation=True,
                              for_ae=True)

    base_model = Model(
        NIST19Net2(train_loader.get_classes_count(), True),
        device
    )
    base_model.load(config['train']['base_model_weights'])

    model = Model(ConfidenceAE2(base_model.model), device)

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
            'NIST19 AE model',
            server=config['visualization']['visdom_server'],
            port=config['visualization']['visdom_port']
        )

        plots.register_scatterplot('train validation loss per_epoch', 'Epochs',
                                   'Loss',
                                   [
                                       'train binary cross entropy',
                                       'validation binary cross entropy'
                                   ])
        callbacks.append(plots)

        callbacks.append(
            VisImageForAE(
                'Image visualisation',
                config['visualization']['visdom_server'],
                config['visualization']['visdom_port'],
                config['train']['image']['every'],
                scale=5
            )
        )

    model.set_callbacks(callbacks)

    start_epoch = 0
    optimizer = torch.optim.Adam(
        model.model.parameters(),
        lr=config['train']['lr']
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        10,
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