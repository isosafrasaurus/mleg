import itertools
import random
from pathlib import Path
from typing import Optional, List, Dict, Any

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import crop
from PIL import Image
from PIL.ImageOps import exif_transpose

class DreamBoothDataset(Dataset):
    """
    A dataset to prepare instance and class images with corresponding prompts for fine-tuning.
    Supports loading from a HuggingFace Dataset or a local image directory.
    """
    def __init__(
        self,
        instance_data_dir: Optional[Path] = None,
        instance_prompt: str = None,
        *,
        dataset_name: Optional[str] = None,
        dataset_config_name: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        image_column: str = "image",
        caption_column: Optional[str] = None,
        repeats: int = 1,
        class_prompt: Optional[str] = None,
        class_data_root: Optional[Path] = None,
        class_num: Optional[int] = None,
        size: int = 512,
        resolution: int = 512,
        center_crop: bool = False,
        random_flip: bool = False,
    ):

        self.size = size
        self.resolution = resolution
        self.center_crop = center_crop
        self.random_flip = random_flip
        self.instance_prompt = instance_prompt
        self.class_prompt = class_prompt

        if dataset_name is not None:

            try:
                from datasets import load_dataset
            except ImportError:
                raise ImportError(
                    "To load a Dataset by name, please install 'datasets': `pip install datasets`"
                )
            dataset = load_dataset(
                dataset_name,
                dataset_config_name,
                cache_dir=str(cache_dir) if cache_dir is not None else None,
            )
            column_names = dataset["train"].column_names
            if image_column not in column_names:
                raise ValueError(
                    f"Image column '{image_column}' not found in dataset columns: {column_names}"
                )
            images = dataset["train"][image_column]

            if caption_column is not None:
                if caption_column not in column_names:
                    raise ValueError(
                        f"Caption column '{caption_column}' not found in dataset columns: {column_names}"
                    )
                captions = dataset["train"][caption_column]
                self.custom_instance_prompts = [c for c in captions for _ in range(repeats)]
            else:
                self.custom_instance_prompts = None

            instance_images = [Image.open(img) if not isinstance(img, Image.Image) else img for img in images]

        else:

            if instance_data_dir is None:
                raise ValueError("Must provide 'instance_data_dir' if not using a HuggingFace Dataset.")
            if not instance_data_dir.exists():
                raise ValueError(f"Instance data directory '{instance_data_dir}' does not exist.")
            instance_images = [Image.open(p) for p in sorted(instance_data_dir.iterdir())]
            self.custom_instance_prompts = None

        self.instance_images = [img for img in instance_images for _ in range(repeats)]

        self.pixel_values: List[torch.Tensor] = []
        train_resize = transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR)
        train_crop = (
            transforms.CenterCrop(self.size) if self.center_crop else transforms.RandomCrop(self.size)
        )
        train_flip = transforms.RandomHorizontalFlip(p=1.0)
        train_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        for image in self.instance_images:
            img = exif_transpose(image)
            if img.mode != "RGB":
                img = img.convert("RGB")
            img = train_resize(img)
            if self.random_flip and random.random() < 0.5:
                img = train_flip(img)
            if self.center_crop:

                y1 = max(0, int((img.height - self.resolution) / 2.0))
                x1 = max(0, int((img.width - self.resolution) / 2.0))
                img = train_crop(img)
            else:
                y1, x1, h, w = train_crop.get_params(img, (self.resolution, self.resolution))
                img = crop(img, y1, x1, h, w)
            tensor_img = train_transforms(img)
            self.pixel_values.append(tensor_img)

        self.num_instance_images = len(self.instance_images)
        self._length = self.num_instance_images

        if class_data_root is not None:
            self.class_data_root = class_data_root
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            class_paths = sorted(self.class_data_root.iterdir())
            if class_num is not None:
                self.num_class_images = min(len(class_paths), class_num)
            else:
                self.num_class_images = len(class_paths)
            self._length = max(self.num_instance_images, self.num_class_images)
        else:
            self.class_data_root = None

        self.image_transforms = transforms.Compose([
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.size) if self.center_crop else transforms.RandomCrop(self.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> Dict[str, Any]:
        example: Dict[str, Any] = {}

        inst = self.pixel_values[index % self.num_instance_images]
        example["instance_images"] = inst
        if self.custom_instance_prompts:
            caption = self.custom_instance_prompts[index % self.num_instance_images]
            example["instance_prompt"] = caption if caption else self.instance_prompt
        else:
            example["instance_prompt"] = self.instance_prompt

        if self.class_data_root:
            class_paths = sorted(self.class_data_root.iterdir())
            class_img = Image.open(class_paths[index % self.num_class_images])
            class_img = exif_transpose(class_img)
            if class_img.mode != "RGB":
                class_img = class_img.convert("RGB")
            example["class_images"] = self.image_transforms(class_img)
            example["class_prompt"] = self.class_prompt

        return example

def collate_fn(examples: List[Dict[str, Any]], with_prior_preservation: bool = False) -> Dict[str, Any]:
    """
    Collate function to form a minibatch. If `with_prior_preservation` is True,
    concatenates class and instance data to enable single forward pass.
    """
    pixel_values = [ex["instance_images"] for ex in examples]
    prompts = [ex["instance_prompt"] for ex in examples]

    if with_prior_preservation:
        pixel_values += [ex["class_images"] for ex in examples]
        prompts += [ex["class_prompt"] for ex in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    return {"pixel_values": pixel_values, "prompts": prompts}

class PromptDataset(Dataset):
    """
    A simple dataset that yields the same prompt `num_samples` times.
    Useful for generating class images in prior preservation.
    """

    def __init__(self, prompt: str, num_samples: int):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return {"prompt": self.prompt, "index": index}