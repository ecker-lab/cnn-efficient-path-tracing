import numpy as np
from os.path import join
from random import shuffle, seed as set_seed
import cv2


class Pathfinder(object):

    def __init__(self, subset, path_length=6, grayscale=False, seed=0, image_size=(150, 150), augmentation=None):
        super().__init__()
        self.grayscale = grayscale
        self.image_size = image_size
        self.augmentation = augmentation

        folders = {
            6: ['curv_baseline'],
            9: ['curv_contour_length_9'],
            14: ['curv_contour_length_14'],
            'all': ['curv_baseline', 'curv_contour_length_9', 'curv_contour_length_14'],
        }[path_length]
        self.base_paths = [join(self.data_path(), folder) for folder in folders]
        splits = list(range(1, 25))

        if seed != 0:
            set_seed(seed)
            shuffle(splits)

        if subset == 'train':
            a_range = splits[0:18]  # range(1, 19)
        elif subset == 'val':
            a_range = splits[18:20]  # range(19, 21)
        elif subset == 'test':
            a_range = splits[20:25]  # range(21, 25)
        else:
            raise ValueError(f'Invalid subset: {subset}')

        self.samples = [(a, b, pos, bp) for a in a_range for b in range(10000) for pos in [True, False] for bp in range(len(self.base_paths))]
        self.sample_ids = list(range(len(self.samples)))

    def __getitem__(self, idx):
        a, b, positive, bp = self.samples[idx]

        suffix = '' if positive else '_neg'
        img = cv2.imread(join(self.base_paths[bp] + suffix, f'imgs/{a}/sample_{b}.png'))
        img = cv2.resize(img, self.image_size, interpolation='nearest')
        img = img.transpose([2, 0, 1]).astype('float32')

        if self.grayscale:
            img = img[:1, :, :]

        if self.augmentation == 'flip':

            if np.random.random() > 0.5:
                img = img[:, ::-1, :]

            if np.random.random() > 0.5:
                img = img[:, :, ::-1]         

        img = img / 255
        img = img - 0.5

        return (img,), (np.array([1.0 if positive else 0.0], dtype='float32'),)