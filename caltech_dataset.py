from torchvision.datasets import VisionDataset

from PIL import Image

import os
import os.path
import sys


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class Caltech(VisionDataset):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        super(Caltech, self).__init__(root, transform=transform, target_transform=target_transform)

        self.split = split # This defines the split you are going to use
                           # (split files are called 'train.txt' and 'test.txt')

        '''
        - Here you should implement the logic for reading the splits files and accessing elements
        - If the RAM size allows it, it is faster to store all data in memory
        - PyTorch Dataset classes use indexes to read elements
        - You should provide a way for the __getitem__ method to access the image-label pair
          through the index
        - Labels should start from 0, so for Caltech you will have lables 0...100 (excluding the background class) 
        '''
        assert(self.split in ['train','test'])
        with open(f'./../{self.split}.txt') as f:
            idx = f.readlines()
            
        # create (0..n, path) tuple list
        
        idx = [x.strip() for x in idx if 'background' not in x.lower()]
        tup_idx_path = zip(range(len(idx)),idx)
        
        # create (0..n, class) tuple list
        
        classes = sorted(list(set([x.split('/')[0].strip() for x in idx])))
        dict_classes = dict(zip(range(len(classes)),classes))
        
        tup_idx_class = dict()
        for i in range(len(idx)):
            tup_idx_class[i] = dict_classes[idx[i].split('/')[0].strip()]
            
        # merge into (path,class) tuple list
        self.split = [(tup_idx_class[i][1],tup_idx_path[i][1]) for i in range(len(idx))]
        

    def __getitem__(self, index):
        '''
        __getitem__ should access an element through its index
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        '''
        path = self.split[index][1]
        img = Image.open(open(os.path.join(self.root,path), 'rb'))
        lab = self.split[index][0]
        image, label = img, lab
                           # Provide a way to access image and label via index
                           # Image should be a PIL Image
                           # label can be int

        # Applies preprocessing when accessing the image
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        '''
        The __len__ method returns the length of the dataset
        It is mandatory, as this is used by several other components
        '''
        length = len(self.split) # Provide a way to get the length (number of elements) of the dataset
        return length
