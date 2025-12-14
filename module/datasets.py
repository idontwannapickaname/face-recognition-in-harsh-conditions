from collections import defaultdict
from typing import Dict, List
import random
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class PairedLightDarkDataset(Dataset):
    def __init__(self, light_map: Dict[int, List[str]], dark_map: Dict[int, List[str]], seed: int = 43):
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        self.light_map = light_map
        self.dark_map = dark_map
        self.person_ids = [pid for pid in self.light_map.keys() if pid in self.dark_map]
        if len(self.person_ids) == 0:
            raise ValueError("No person IDs appear in both light and dark maps.")
        random.seed(seed)

    def __len__(self):
        return len(self.person_ids)

    def __getitem__(self, idx):
        weights = [0, 1, 1.2]
        choice = random.choices(['pull_2_dark', 'push_2_dark', 'pull_light_dark'], weights)
        if choice == ['pull_light_dark']:
            pid = random.choice(self.person_ids)
            p1 = random.choice(self.light_map[pid])
            p2 = random.choice(self.dark_map[pid])
            target = torch.tensor(1.0)
        elif choice == ['pull_2_dark']:
            pid = random.choice(self.person_ids)
            p1, p2 = random.sample(self.dark_map[pid], 2)
            target = torch.tensor(1.0)
        else:
            pid1, pid2 = random.sample(self.person_ids, 2)
            p1 = random.choice(self.dark_map[pid1])
            p2 = random.choice(self.dark_map[pid2])
            target = torch.tensor(-1.0)
        img1 = cv2.cvtColor(cv2.imread(p1), cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(cv2.imread(p2), cv2.COLOR_BGR2RGB)
        img1 = self.transform(img1)
        img2 = self.transform(img2)
        return img1, img2, target
