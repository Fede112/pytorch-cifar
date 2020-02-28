import numpy as np
from PIL import Image

from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler

class SiameseCIFAR(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, cifar_dataset, num_pairs = None):
        self.cifar_dataset = cifar_dataset

        self.train = self.cifar_dataset.train
        self.transform = self.cifar_dataset.transform

        # if self.train:
        self.targets = self.cifar_dataset.targets
        self.data = self.cifar_dataset.data
        self.labels_set = set(cifar_dataset.class_to_idx.values())
        self.label_to_indices = {label: np.where(np.array(self.targets) == label)[0]
                                 for label in self.labels_set}

        self.num_pairs = len(self.cifar_dataset)
        if num_pairs:
            self.num_pairs = num_pairs

        print (self.num_pairs)
        print (self.labels_set)
        print (self.label_to_indices)
        random_state = np.random.RandomState(29)

        positive_pairs = [[i, random_state.choice(self.label_to_indices[self.targets[i]]), 1]
                          for i in range(0, self.num_pairs, 2)]

        negative_pairs = [[i, 
                            random_state.choice(self.label_to_indices[ 
                                np.random.choice( list( self.labels_set - set([self.targets[i]]) )   ) ]),
                           0]
                          for i in range(1, self.num_pairs, 2)]
        
        self.pairs = positive_pairs + negative_pairs



    def __getitem__(self, index):
        img1 = self.data[self.pairs[index][0]]
        img2 = self.data[self.pairs[index][1]]
        target = self.pairs[index][2]
        print(img1.shape)
        img1 = Image.fromarray(img1)
        img2 = Image.fromarray(img2)
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return (img1, img2), target

    def __len__(self):
        return len(self.num_pairs)