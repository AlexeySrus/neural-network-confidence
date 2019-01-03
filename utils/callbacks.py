import os
import torch
import numpy as np
from visdom import Visdom


class SaveModelPerEpoch:
    def __init__(self, path, save_step=1):
        self.path = path
        self.step=save_step

        if not os.path.isdir(path):
            os.makedirs(path)

    def per_batch(self, args):
        pass

    def per_epoch(self, args):
        if args['n'] % self.step == 0:
            args['model'].save(
                os.path.join(self.path, 'model-{}.trh'.format(args['n']))
            )


class SaveOptimizerPerEpoch:
    def __init__(self, path, save_step=1):
        self.path = path
        self.step=save_step

        if not os.path.isdir(path):
            os.makedirs(path)

    def per_batch(self, args):
        pass

    def per_epoch(self, args):
        if args['n'] % self.step == 0:
            torch.save(args['optimize_state'], (
                os.path.join(
                    self.path,
                    'optimize_state-{}.trh'.format(args['n'])
                )
            ))


class VisPlot(object):
    def __init__(self, title, server='https://localhost', port=8080):
        self.viz = Visdom(server=server, port=port)
        self.windows = {}
        self.title = title

    def register_scatterplot(self, name, xlabel, ylabel, legend=None):
        options = dict(title=self.title, markersize=5,
                        xlabel=xlabel, ylabel=ylabel) if legend is None \
                       else dict(title=self.title, markersize=5,
                        xlabel=xlabel, ylabel=ylabel,
                        legend=legend)

        self.windows[name] = [None, [], [], [], options]

    def update_scatterplot(self, name, x, y1, y2=None, avg=100):

        if y2 is None:
            self.windows[name][0] = self.viz.line(
                np.convolve(y1, (1 / avg,) * avg, mode='valid'),
                np.convolve(x, (1 / avg,) * avg, mode='valid'),
                win=self.windows[name][0],
                opts=self.windows[name][4]
            )
        else:
            self.windows[name][0] = self.viz.line(
                np.transpose(
                    np.array([np.convolve(y1, (1 / avg,) * avg, mode='valid'),
                              np.convolve(y2, (1 / avg,) * avg, mode='valid')])
                ),
                np.convolve(x, (1 / avg,) * avg, mode='valid'),
                win=self.windows[name][0],
                opts=self.windows[name][4]
            )

    def per_batch(self, args, keyward='per_batch'):
        for win in self.windows.keys():
            if keyward in win:
                if 'train' in win:
                    self.windows[win][1].append(args['n'])
                    self.windows[win][2].append(args['loss'])
                    self.update_scatterplot(
                        win,
                        self.windows[win][1],
                        self.windows[win][2]
                    )

                if 'validation' in win:
                    self.windows[win][1].append(args['n'])
                    self.windows[win][2].append(args['val loss'])
                    self.update_scatterplot(
                        win,
                        self.windows[win][1],
                        self.windows[win][2]
                    )

    def per_epoch(self, args, keyward='per_epoch'):
        for win in self.windows.keys():
            if keyward in win:
                if 'train' in win:
                    self.windows[win][1].append(args['n'])
                    self.windows[win][2].append(args['loss'])
                    self.update_scatterplot(
                        win,
                        self.windows[win][1],
                        self.windows[win][2]
                    )

                if 'validation' in win:
                    self.windows[win][1].append(args['n'])
                    self.windows[win][2].append(args['val loss'])
                    self.update_scatterplot(
                        win,
                        self.windows[win][1],
                        self.windows[win][2]
                    )

                if 'train' in win and 'validation' in win:
                    self.windows[win][1].append(args['n'])
                    self.windows[win][2].append(args['loss'])
                    self.windows[win][3].append(args['val loss'])
                    self.update_scatterplot(
                        win,
                        self.windows[win][1],
                        self.windows[win][2],
                        self.windows[win][3]
                    )
