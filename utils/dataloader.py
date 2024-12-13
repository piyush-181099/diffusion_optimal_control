from glob import glob
from PIL import Image
from typing import Callable, Optional
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset, CIFAR10


__DATASET__ = {}

def register_dataset(name: str):
    def wrapper(cls):
        if __DATASET__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __DATASET__[name] = cls
        return cls
    return wrapper


def get_dataset(name: str, root: str, **kwargs):
    if __DATASET__.get(name, None) is None:
        raise NameError(f"Dataset {name} is not defined.")
    return __DATASET__[name](root=root, **kwargs)


def get_dataloader(dataset: VisionDataset,
                   batch_size: int, 
                   num_workers: int, 
                   train: bool,
                   shuffle=None,
                  ):
    shuffle = train if shuffle is None else shuffle
    dataloader = DataLoader(dataset, 
                            batch_size, 
                            shuffle=shuffle, 
                            num_workers=num_workers, 
                            drop_last=train)
    return dataloader


@register_dataset(name='ffhq')
class FFHQDataset(VisionDataset):
    def __init__(self, root: str, transforms: Optional[Callable]=None):
        super().__init__(root, transforms)
        
        fpaths = sorted(glob(root + '/**/*.png', recursive=True) + glob(root + '/**/*.jpg', recursive=True))
        if 'ffhq' in root:
            self.fpaths = []
            for fpath in fpaths:
                fnum = int(fpath.split('/')[-1].split('.')[0])
                if fnum >= 69000:
                    self.fpaths.append(fpath)
        else:
            self.fpaths = fpaths
            
        assert len(self.fpaths) > 0, "File list is empty. Check the root."

    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, index: int):
        fpath = self.fpaths[index]
        img = Image.open(fpath).convert('RGB')
        
        if self.transforms is not None:
            img = self.transforms(img)
        
        return img
    
@register_dataset(name='cifar')
class CIFARDataset(VisionDataset):
    def __init__(self, root: str, transforms: Optional[Callable]=None):
        super().__init__(root, transforms=transforms)
        
        self.dataset = CIFAR10(root, train=False)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        img, label = self.dataset[index]
        
        if self.transforms is not None:
            img = self.transforms(img)
        
        return img