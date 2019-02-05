import cv2
import os
import argparse
from tqdm import tqdm
from multiprocessing import Pool

dirs = ['4a', '4b', '4c', '4d', '4e', '4f', '5a', '6a', '6b', '6c', '6d', '6e', '6f', '7a', '30', '31', '32', '33',
            '34', '35', '36', '37', '38', '39', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50',
            '51', '52', '53', '54', '55', '56', '57', '58', '59', '61', '62', '63', '64', '65', '66', '67', '68', '69',
            '70',
            '71', '72', '73', '74', '75', '76', '77', '78', '79']

classes = ['J', 'K', 'L', 'M', 'N', 'O', 'Z', 'j', 'k', 'l', 'm', 'n', 'o', 'z', '0', '1', '2', '3',
               '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'P',
               'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'p',
               'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y']


def flatten_list(lst):
    res = []
    for l in lst:
        for e in l:
            res.append(e)
    return res


def parse_arguments():
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('filepath',
                            help='Path to created data file.')
    return arg_parser.parse_args()


def generate_imgs_tuple(path):
    supdirs = list(set(['hsf_{}'.format(i) for i in range(8)]) - set(['hsf_5']))

    classes_dirs_tuple = {
        c: [path + '/by_class/' + d + '/' + s + '/' for s in supdirs] +
           [path + '/by_class/' + d + '/train_' + d + '/']
        for c, d in zip(classes, dirs)
    }

    classes_imgs_tuple = {
        c: flatten_list(
            [[d + img_name for img_name in os.listdir(d)] for d in drs_list])
        for c, drs_list in classes_dirs_tuple.items()
    }

    return classes_imgs_tuple


def generate_paths(path):
    classes_imgs_tuple = generate_imgs_tuple(path)

    _label_img_store = [(label, img) for label in classes for img in
                       classes_imgs_tuple[label]]

    return _label_img_store


def get_actual_area(img, threshold=200):
    x0, x1 = 0, 0
    y0, y1 = 0, 0

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] < 200:
                if y0 == 0:
                    y0 = i
                y1 = i

    for j in range(img.shape[1]):
        for i in range(img.shape[0]):
            if img[i, j] < threshold:
                if x0 == 0:
                    x0 = j
                x1 = j

    width = x1 - x0
    height = y1 - y0

    d = width - height

    if d < 0:
        x0 += d // 2
        x1 -= d // 2
    else:
        y0 -= d // 2
        y1 += d // 2

    width = x1 - x0
    height = y1 - y0

    x1 -= width - height

    return img[y0:y1 + 1, x0:x1 + 1]


def imap_unordered_bar(func, args, n_processes=8):
    p = Pool(n_processes)
    res_list = []
    with tqdm(total = len(args)) as pbar:
        for i, res in tqdm(enumerate(p.imap_unordered(func, args))):
            pbar.update()
            res_list.append(res)
    pbar.close()
    p.close()
    p.join()
    return res_list


def argument_parser():
    arg_pars = argparse.ArgumentParser()
    arg_pars.add_argument('--path',
                          required=True,
                          type=str)
    arg_pars.add_argument('--njobs',
                          required=False,
                          type=int,
                          default=8)
    return arg_pars.parse_args()


def change_data_image(img_path):
    img = cv2.imread(img_path[1], 0)

    if img is None:
        return False

    img = get_actual_area(img)

    cv2.imwrite(img_path[1], img)

    return True


if __name__ == '__main__':
    args = argument_parser()

    data = generate_paths(args.path)

    print(data[0])

    res = imap_unordered_bar(change_data_image, data, args.njobs)

    print('Update {} from {}'.format(sum(res), len(data)))