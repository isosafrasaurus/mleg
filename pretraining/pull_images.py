import json
import requests
from io import BytesIO
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, json_file, transform=None):
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        self.urls = [item["url"] for item in self.data["images"]]
        self.transform = transform
    def __len__(self):
        return len(self.urls)
    def __getitem__(self, idx):
        url = self.urls[idx]
        response = requests.get(url)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
dataset = ImageDataset("images.json", transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

for batch in dataloader:
    print(batch.size())
    break
