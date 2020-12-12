import numpy as np
import cv2
from os.path import join
from random import shuffle, seed as set_seed


class CABC(object):

    def __init__(self, subset, difficulty=0, grayscale=False, seed=0, augmentation=None):
        super().__init__()
        self.grayscale = grayscale
        self.difficulty = difficulty
        self.augmentation = augmentation

        folders = {
            0: ['baseline-/media/data_cifs/cluttered_nist3/baseline-/'],
            1: ['ix1-/media/data_cifs/cluttered_nist3/ix1-/'],
            3: ['ix2/media/data_cifs/cluttered_nist3/ix2/'],
            'all': [
                'baseline-/media/data_cifs/cluttered_nist3/baseline-/', 
                'ix1-/media/data_cifs/cluttered_nist3/ix1-/',
                'ix2/media/data_cifs/cluttered_nist3/ix2/'
            ],
        }[difficulty]
        self.base_paths = [join(self.data_path(), folder) for folder in folders]
        splits = list(range(1, 51))

        if seed != 0:
            set_seed(seed)
            shuffle(splits)

        if subset == 'train':
            a_range = splits[0:36]  # range(1, 19)
        elif subset == 'val':
            a_range = splits[36:40] # range(19, 21)
        elif subset == 'test':
            a_range = splits[40:51] # range(21, 25)
        else:
            raise ValueError(f'Invalid subset: {subset}')

        self.samples = [(a, b, bp) for a in a_range for b in range(4000) for bp in range(len(self.base_paths))]
        self.labels = [np.load(join(self.base_paths[0], f'metadata/{i}.npy'))[:,[0,2,4]].astype('U13') for i in range(1,51)]
        self.sample_ids = list(range(len(self.samples)))

    def __getitem__(self, idx):

        g, sample, bp = self.samples[idx]
        _, _, positive = self.labels[g-1][sample]

        img = cv2.imread(join(self.base_paths[0], f'imgs/{g}/sample_{sample}.png'))
        img = cv2.resize(img, (150, 150), channel_dim=0, interpolation='nearest')
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

        return (img,), (np.array([1.0] if positive=='1' else [0.0], dtype='float32'),)
