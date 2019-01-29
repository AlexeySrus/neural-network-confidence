import torch


def classification_with_confidence(x, basic_net, ae_net):
    y1 = basic_net(x)[0]
    x_gen = ae_net(x)
    y2 = basic_net(x_gen)[0]
    argmax_y1 = torch.argmax(y1, dim=1)
    argmax_y2 = torch.argmax(y2, dim=1)
    return y1, y2, (1 - (y1 - y2).abs().sum() / 2).detach().to('cpu').numpy(), x_gen
