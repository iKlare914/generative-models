from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch as th

class CifarDataset(Dataset):
    def __init__(self, name, split, shuffle=True, image_size=None, cache_dir=None):
        super().__init__()
        self.data = load_dataset(name, split=split, cache_dir=cache_dir)
        self.image_size = image_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index]['img'].convert('RGB')
        if self.image_size is not None and image.size != (self.image_size, self.image_size):
            image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        label = self.data[index]['label']
        return self.normalize_img(image), label

    def normalize_img(self, img):
        """
        Convert uint8 data to [-1 ,1] by data / 127.5 - 1
        """
        img = th.from_numpy(np.array(img, dtype=np.float32))
        img = img.permute(2, 0, 1).contiguous()
        img = img / 127.5 - 1 # convert to [-1, 1]
        return img

def getCifarLoader(repo_name, split, batch_size=8, shuffle=True, num_workers=4, image_size=None, cache_dir=None):
    dataset = CifarDataset(repo_name, split, shuffle, image_size=image_size, cache_dir=cache_dir)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    return loader

if __name__ == '__main__':
    repo_name = "uoft-cs/cifar10"
    loader = getCifarLoader(repo_name, 'train')
    image, label = next(iter(loader))
    print(image, label)