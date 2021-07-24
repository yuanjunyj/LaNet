import numpy as np
import random
from torch.utils.data.dataset import Dataset

from cutmix.utils import onehot, rand_bbox


class CutMix(Dataset):
    def __init__(self, dataset, num_class, num_mix=1, beta=1., prob=1.0, use_mixup=False, smoothing=0.0):
        self.dataset = dataset
        self.num_class = num_class
        self.num_mix = num_mix
        self.beta = beta
        self.prob = prob
        self.use_mixup = use_mixup
        self.label_smoothing = smoothing

    def __getitem__(self, index):
        img, lb = self.dataset[index]
        lb_onehot = onehot(self.num_class, lb, self.label_smoothing)

        if not self.use_mixup or np.random.rand(1) < 0.5:
            for _ in range(self.num_mix):
                r = np.random.rand(1)
                if self.beta <= 0 or r > self.prob:
                    continue

                # generate mixed sample
                lam = np.random.beta(self.beta, self.beta)
                rand_index = random.choice(range(len(self)))

                img2, lb2 = self.dataset[rand_index]
                lb2_onehot = onehot(self.num_class, lb2, self.label_smoothing)

                bbx1, bby1, bbx2, bby2 = rand_bbox(img.size(), lam)
                img[:, bbx1:bbx2, bby1:bby2] = img2[:, bbx1:bbx2, bby1:bby2]
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (img.size()[-1] * img.size()[-2]))
                lb_onehot = lb_onehot * lam + lb2_onehot * (1. - lam)
        else:
            for _ in range(self.num_mix):
                r = np.random.rand(1)
                if self.beta <= 0 or r > self.prob:
                    continue

                # generate mixed sample
                lam = np.random.beta(self.beta, self.beta)
                rand_index = random.choice(range(len(self)))

                img2, lb2 = self.dataset[rand_index]
                lb2_onehot = onehot(self.num_class, lb2, self.label_smoothing)

                img = img * lam + img2 * (1. - lam)
                lb_onehot = lb_onehot * lam + lb2_onehot * (1. - lam)

        return img, lb_onehot

    def __len__(self):
        return len(self.dataset)
