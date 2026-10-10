from datasets import Dataset as HFDataset, load_dataset, load_from_disk
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch as th
from generative_model.dataset import image_transforms
from generative_model.dataset.text_encoder import CLIPTextEncoder, CIFAR10_LABELS, CIFAR10_PROMPT_TEMPLATES

class CifarDataset(Dataset):
    def __init__(self, name, split, image_size=None, cache_dir=None):
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

class ImagenetDataset(Dataset):
    """ImageNet images and labels, with an optional persistent resize cache.

    image_size is an integer for square images or a (height, width) tuple.
    It is required only when resize=True. cache_dir controls the Hugging
    Face source cache; resized datasets live under image_transforms.DATASET_DIR.
    Initialization uses an existing resize cache without accessing the Hub.
    Only a missing resize cache (or resize=False) loads the source dataset.
    Resize caches are named <repo_name>-<H>x<W>-<split> and store one Dataset.
    Use split=None or split="all" to combine all source splits into one Dataset.
    """

    def __init__(
        self, name="clane9/imagenet-100", split="train", resize=False,
        image_size=None, cache_dir=None,
    ):
        super().__init__()
        split = "all" if split is None else split
        if not isinstance(split, str) or not split:
            raise ValueError("split must be None or a nonempty string.")
        self.image_size = image_size
        self.resize = resize
        resize_size = (image_size, image_size) if type(image_size) is int else image_size
        if resize and (
            not isinstance(resize_size, tuple) or len(resize_size) != 2
            or any(type(size) is not int or size <= 0 for size in resize_size)
        ):
            raise ValueError("image_size must be a positive integer or a (height, width) tuple when resize=True.")

        resized_dir = (
            image_transforms.get_resized_dataset_dir(name, resize_size, split)
            if resize else None
        )
        if not resize or not resized_dir.exists():
            self.data = load_dataset(name, split=split, cache_dir=cache_dir)
            if not isinstance(self.data, HFDataset):
                raise TypeError("load_dataset must return a single Hugging Face Dataset.")
            if resize:
                image_transforms.resize_image_dataset(
                    self.data, resize_size, repo_name=name, split=split,
                )
        if resize:
            cached = load_from_disk(resized_dir)
            if not isinstance(cached, HFDataset):
                raise TypeError(f"Resize cache {resized_dir} must contain a single Dataset.")
            # 'all' may be recorded as train+validation by Hugging Face.
            # Its directory suffix identifies the combined cache without
            # loading the source dataset just to discover its split names.
            if split != "all" and cached.split != split:
                raise ValueError(f"Resize cache {resized_dir} does not match split {split!r}.")
            self.data = cached

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        image = sample["image"].convert("RGB")
        return self.normalize_img(image), sample["label"]

    def normalize_img(self, img):
        """Convert an image to a contiguous float32 [3, H, W] tensor in [-1, 1]."""
        img = th.from_numpy(np.array(img, dtype=np.float32))
        return img.permute(2, 0, 1).contiguous() / 127.5 - 1


class CifarCLIPDataset(CifarDataset):
    """Return CIFAR images and precomputed conditioning for random class prompts."""

    def __init__(self, name, split, image_size=None, cache_dir=None, *, text_encoder=None, max_length=16, label_drop_rate=0.2):
        super().__init__(name, split, image_size=image_size, cache_dir=cache_dir)
        if not 0 <= label_drop_rate <= 1:
            raise ValueError("label_drop_rate must be between 0 and 1.")
        self.label_drop_rate = label_drop_rate
        self.label2text = CIFAR10_LABELS.copy()
        self.prompt_template = list(CIFAR10_PROMPT_TEMPLATES)
        self.max_length = max_length
        prompts = [""] + [template.format(label) for label in self.label2text.values() for template in self.prompt_template]
        encoder = text_encoder if text_encoder is not None else CLIPTextEncoder()
        context, mask = encoder.encode_prompts(prompts, max_length=max_length)
        self.hidden_state_table = dict(zip(prompts, context.unbind(0)))
        self.attention_mask_table = dict(zip(prompts, mask.unbind(0)))
        # Keep only CPU tensors in the dataset, not the encoder or its model,
        # so DataLoader workers do not receive a copy of CLIP.

    def __getitem__(self, index: int) -> tuple[th.Tensor, tuple[th.Tensor, th.Tensor]]:
        """Return an image and cached features for a randomly selected class prompt.

        Args:
            index: Index of the dataset sample.

        Returns:
            A tuple (image, (context, attention_mask)) containing:
                image: CPU float32 tensor of shape [3, H, W], scaled to [-1, 1].
                context: CPU text features of shape [max_length, 512] for default CLIP.
                attention_mask: CPU bool tensor of shape [max_length], True for valid tokens.
            Text encoding runs only at initialization. With probability label_drop_rate,
            return cached empty-text features and their BOS/EOS mask instead.

        Example:
            image, (context, attention_mask) = dataset[index]
        """
        image, label = super().__getitem__(index)
        prompt = "" if np.random.random() < self.label_drop_rate else self.get_prompt(label)
        return image, (self.hidden_state_table[prompt], self.attention_mask_table[prompt])

    def get_prompt(self, label):
        """
        Get prompt text for a given CIFAR-10 label (0 through 9).
        """
        text = self.label2text[label]
        prompt = np.random.choice(self.prompt_template).format(text)
        return prompt

def getCifarLoader(repo_name, split, batch_size=8, shuffle=True, num_workers=8, image_size=None, cache_dir=None, drop_last=True):
    dataset = CifarDataset(repo_name, split, image_size=image_size, cache_dir=cache_dir)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=True
    )
    return loader

def getCifarCLIPLoader(repo_name, split, batch_size=8, shuffle=True, num_workers=4, image_size=None, cache_dir=None, drop_last=True, *, text_encoder=None, max_length=16, label_drop_rate=0.2):
    dataset = CifarCLIPDataset(
        repo_name, split, image_size=image_size, cache_dir=cache_dir,
        text_encoder=text_encoder, max_length=max_length, label_drop_rate=label_drop_rate,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last
    )
    return loader

if __name__ == '__main__':
    repo_name = "uoft-cs/cifar10"
    loader = getCifarLoader(repo_name, 'train')
    image, label = next(iter(loader))
    print(image.shape, label.shape)
    loader_clip = getCifarCLIPLoader(repo_name, 'train')
    image, (context, attention_mask) = next(iter(loader_clip))
    print(image.shape, context.shape, attention_mask.shape)
