import torch
import argparse
import tqdm
from model.cifar10_resnet import ResNet18, ConfidenceAE
from model.model import Model
from utils.callbacks import VisImageForAE
from utils.loaders import load_cifar10, get_loaders
from utils.confidence_prediction import classification_with_confidence
from sklearn.metrics import accuracy_score
import yaml


def parse_args():
    parser = argparse.ArgumentParser(description='AE test script')
    parser.add_argument('--config', required=True, type=str,
                        help='Path to configuration yml file.')
    parser.add_argument('--confidence', required=False, type=float,
                        default=0.5)
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    return parser.parse_args()


def main():
    args = parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    base_model = Model(ResNet18(), device)
    base_model.load(config['train']['base_model_weights'])
    ae_model = Model(ConfidenceAE(base_model.model), device)
    ae_model.load(config['train']['ae_model_weights'])

    draw = VisImageForAE(
        'Image test visualisation',
        config['visualization']['visdom_server'],
        config['visualization']['visdom_port'],
        1
    )

    base_model.model.eval()
    ae_model.model.eval()

    train_loader, val_loader = get_loaders(load_cifar10())

    k = 1
    y = []
    y1 = []
    y_by_ae = []

    angle = {
        0: 90,
        1: 180,
        2: 270
    }

    for i in tqdm.tqdm(range(1000)):
        x, y_true = val_loader[i]

        x = torch.FloatTensor(x).to(device).unsqueeze(0)

        x = torch.clamp(
            x + torch.FloatTensor(1, 3, 32, 32).to(device).normal_(0, 0.0),
            0, 1
        )

        #x = rotate_tensor(x.squeeze(0), angle[i % 3]).unsqueeze(0)

        y_pred1, y_pred2, conf, x_gen = classification_with_confidence(
            x,
            base_model.model,
            ae_model.model
        )

        y_pred1 = y_pred1.detach().to('cpu').numpy()
        y_pred2 = y_pred2.detach().to('cpu').numpy()

        if False and conf < args.confidence:
            print(
                'i:', k,
                'y_true:', y_true.argmax(), 'y_pred1:',  y_pred1.argmax(),
                'y_pred2:', y_pred2.argmax(), 'confidence:', conf,
                'result:', conf > 0.8
            )

            draw.add_window(k)
            draw.per_batch({
                'y_true': x,
                'y_pred': x_gen
            }, k)

            k += 1

        y.append(y_true.argmax())
        y1.append(y_pred1.argmax())
        y_by_ae.append(
            y_pred1.argmax() if conf > args.confidence else y_pred2.argmax()
        )

    print('Default accuracy:', accuracy_score(y, y1))
    print('AE accuracy:', accuracy_score(y, y_by_ae))

    y = []
    y1 = []

    for i in tqdm.tqdm(range(100 + 0*len(val_loader))):
        x, y_true = val_loader[i]

        x = torch.FloatTensor(x).to(device).unsqueeze(0)

        x = torch.clamp(
            x + torch.FloatTensor(1, 3, 32, 32).to(device).normal_(0, 0.0),
            0, 1
        )

        y_pred1, y_pred2, conf, x_gen = classification_with_confidence(
            x,
            base_model.model,
            ae_model.model
        )

        y_pred1 = y_pred1.detach().to('cpu').numpy()

        if conf > args.confidence:
            y.append(y_true.argmax())
            y1.append(y_pred1.argmax())

    print('Accuracy by confidence:', accuracy_score(y, y1))

    print('Drop elements rate:', (len(val_loader) - len(y)) / len(val_loader))


if __name__ == '__main__':
    main()
