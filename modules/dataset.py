import os

import torch
from torch.utils.data import Dataset
from skimage import io

from utils import generate_phoc_vector, generate_phos_vector

import pandas as pd
import numpy as np


class phosc_dataset(Dataset):
    def __init__(self, csvfile, root_dir, transform=None, calc_phosc=True):
        self.root_dir = root_dir
        self.transform = transform
        self.df_all = pd.read_csv(csvfile, usecols=["Image", "Word"])
        self.df_all["phos"] = self.df_all['Word'].apply(generate_phos_vector)
        self.df_all["phoc"] = self.df_all['Word'].apply(generate_phoc_vector)

        if calc_phosc:
            self.df_all['phosc'] = self.df_all['phos']
            for i in range(len(self.df_all["phos"])):
                self.df_all['phosc'][i] = np.concatenate((self.df_all["phos"][i], self.df_all["phoc"][i]))

        print(len(self.df_all['phosc'][0]))
        print(self.df_all)
        # len phos-vector: 165
        # len phoc-vector: 604

        # Fill in your code here. You will populate self.df_all
        # it should be pandas df with ["Image", "Word", "phos", "phoc", "phosc"] columns
        # containing file name, word label, phoc, phoc, phosc features vector in each row
        # phosc features vector can be created combining generate_phos_vector, generate_phoc_vector
        # Note: How to use phoc, phoc or phosc of the df in a batch is up to you.
        # in the __getitem__ below the phosc vector is used in the batches.

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.df_all.iloc[index, 0])
        image = io.imread(img_path)

        y = torch.tensor(self.df_all.iloc[index, len(self.df_all.columns) - 1])

        if self.transform:
            image = self.transform(image)

        return image.float(), y.float(), self.df_all.iloc[index, 1]

    def __len__(self):
        return len(self.df_all)


if __name__ == '__main__':
    from torchvision.transforms import transforms

    dataset = phosc_dataset('image_data/IAM_test_unseen.csv', '../image_data/IAM_test', transform=transforms.ToTensor())

    print(dataset.df_all)

    print(dataset.__getitem__(0))
