import torch.utils.data as data
from PIL import Image
import os
import torch as torch
import os.path
import torchvision.transforms as transforms

class ImdbDataset(data.Dataset):
    def __init__(self, data, image_folder):
        self.image_folder = image_folder
        self.data = data

    def __getitem__(self, index):
        item = self.data[index]
        img = Image.open(os.path.join(self.image_folder, item[27])).convert('RGB')
        img_transformator = self._image_transform()
        img = img_transformator(img)
        features = [[item[0]]+ [item[2]]+ item[6:27]][0]
        features = [int(x) for x in features]
        # image, item, target
        return img, torch.FloatTensor(features), int(item[1])

    def __len__(self):
        return len(self.data)

    def _image_transform(self):
        transform = transforms.Compose([
            transforms.Scale(152),
            transforms.RandomCrop(128),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
        return transform
