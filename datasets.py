import numpy as np
from PIL import Image

from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler

class SiameseCIFAR(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, cifar_dataset, pairs_per_label = None):
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
        random_state = np.random.RandomState(29)

        self.pairs = []
        if pairs_per_label:
            self.num_pairs = len(self.labels_set)*pairs_per_label
            for label in self.labels_set:
                
                positive_pairs = [[i, random_state.choice(self.label_to_indices[self.targets[i]]), 1]
                                    for i in self.label_to_indices[label][0:pairs_per_label:2]]

                negative_pairs = [[i, 
                            random_state.choice(self.label_to_indices[ 
                                random_state.choice( list( self.labels_set - set([self.targets[i]]) )   ) ]),
                           0]
                          for i in self.label_to_indices[label][1:pairs_per_label:2]]
                
                self.pairs += positive_pairs + negative_pairs    
            return None

        positive_pairs = [[i, random_state.choice(self.label_to_indices[self.targets[i]]), 1]
                          for i in range(0, self.num_pairs, 2)]

        negative_pairs = [[i, 
                            random_state.choice(self.label_to_indices[ 
                                random_state.choice( list( self.labels_set - set([self.targets[i]]) )   ) ]),
                           0]
                          for i in range(1, self.num_pairs, 2)]
        

        self.pairs = positive_pairs + negative_pairs
        return None 
        # pair_list = []
        # for pair in self.pairs:
        #     pair_list.append(self.targets[pair[0]])
        # unique, counts = np.unique(np.array(pair_list), return_counts=True)
        # print( dict(zip(unique, counts)) )
        

    def __getitem__(self, index):
        img1 = self.data[self.pairs[index][0]]
        img2 = self.data[self.pairs[index][1]]
        target = self.pairs[index][2]
        img1 = Image.fromarray(img1)
        img2 = Image.fromarray(img2)
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return (img1, img2), target

    def __len__(self):
        return self.num_pairs