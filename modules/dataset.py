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

        # We create a new DataFrame by using the first two columns of the CSV-file.
        self.df_all = pd.read_csv(csvfile, usecols=["Image", "Word"])

        # Now we add the PHOS and PHOC feature vectors, this is done by using the generate_vector functions
        #   with each word as input
        self.df_all["phos"] = self.df_all['Word'].apply(generate_phos_vector)
        self.df_all["phoc"] = self.df_all['Word'].apply(generate_phoc_vector)

        # For the PHOSC vector we concateante the PHOS and PHOC vectors. The PHOSC vector will be used again for the
        #   loss functions
        if calc_phosc:
            self.df_all['phosc'] = ''
            for i in range(len(self.df_all["phos"])):
                self.df_all['phosc'][i] = np.concatenate((self.df_all["phos"][i], self.df_all["phoc"][i]))

        print("length of phos vector: ", len(self.df_all['phos'][0]))
        print("length of phoc vector: ", len(self.df_all['phoc'][0]))
        print("length of phosc vector: ", len(self.df_all['phosc'][0]))

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
