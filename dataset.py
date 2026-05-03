
import os
import cv2
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class BrainMRIDataset(Dataset):
    def __init__(self, image_dir, mask_dir, split="train"):
        self.image_dir   = image_dir
        self.mask_dir    = mask_dir
        self.image_files = sorted(os.listdir(image_dir))
        self.transform   = self._get_transforms(split)

    def _get_transforms(self, split):
        if split == "train":
            return A.Compose([
                A.Resize(256, 256),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.RandomRotate90(p=0.5),
                A.ElasticTransform(alpha=1, sigma=50, p=0.3),
                A.GridDistortion(p=0.2),
                A.RandomBrightnessContrast(p=0.3),
                A.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
        else:
            return A.Compose([
                A.Resize(256, 256),
                A.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        fname = self.image_files[idx]

        img  = cv2.imread(os.path.join(self.image_dir, fname))
        img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(self.mask_dir, fname), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)   # binary: 0 or 1

        aug = self.transform(image=img, mask=mask)
        return aug["image"], aug["mask"].unsqueeze(0)
