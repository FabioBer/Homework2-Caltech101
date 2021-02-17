from torchvision.datasets import VisionDataset

from PIL import Image

import os
import os.path
import sys
import math


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class Caltech(VisionDataset):
    def __init__(self, root, split=None, transform=None, target_transform=None):
        super(Caltech, self).__init__(root, transform=transform, target_transform=target_transform)

        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        
        imgs = list()
        if split is None:
            with open(os.path.join(root,'train.txt')) as f:
                imgs = f.readlines()
            with open(os.path.join(root,'test.txt')) as f:
                for line in f:
                    imgs.append(line)
        elif split=='train':
            with open(os.path.join(root,'train.txt')) as f:
                imgs = f.readlines()
        elif split=='test':
            with open(os.path.join(root,'test.txt')) as f:
                imgs = f.readlines()
            
        self.images = list()
        for img in imgs:
            if not str(img).startswith("BACKGROUND_Google/"):
                self.images.append(img)
        
        self.length = len(self.images)
        
        self.classes = os.listdir(self.root+'101_ObjectCategories')
        self.classes.pop(self.classes.index('BACKGROUND_Google'))
        
        self.tuple_classes = list() 
        for i in range(len(self.classes)):
            self.tuple_classes.append((i,self.classes[i]))
            
        self.couples = list()
        for img in self.images:
            s = img.split('/')[0]
            i = self.classes.index(s)
            self.couples.append((i,img))
            

    def __getitem__(self, index):
            
        label,image = self.couples[index]
        
        image = pil_loader(self.root+'101_ObjectCategories/'+image.strip())

        # Applies preprocessing when accessing the image
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        
        length = self.length
        return length
