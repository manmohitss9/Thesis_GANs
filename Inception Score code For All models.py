#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from scipy.stats import entropy
import os
from PIL import Image


class SingleFolderDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image


def inception_score(dataloader, inception_model, device):
    inception_model.eval()
    preds = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            pred = inception_model(batch)  

            
            if pred.dim() > 2:
                pred = F.adaptive_avg_pool2d(pred, (1, 1))
                pred = pred.squeeze(-1).squeeze(-1)  

            preds.append(F.softmax(pred, dim=1).cpu().numpy())
    
    preds = np.concatenate(preds, axis=0)
    py = np.mean(preds, axis=0)
    
    scores = []
    eps = 1e-8  
    
    for i in range(preds.shape[0]):
        pyx = preds[i] + eps
        scores.append(entropy(pyx, py))
    
    return np.exp(np.mean(scores))


transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  
])


synthetic_images_folder = "/media/orin/USBstore/WGAN1_FID/WGAN1_generated_images"
synthetic_dataset = SingleFolderDataset(image_folder=synthetic_images_folder, transform=transform)
synthetic_loader = DataLoader(synthetic_dataset, batch_size=32, shuffle=False)

# InceptionV3 model
inception_model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
inception_model.fc = torch.nn.Identity()  
inception_model.eval()
inception_model.to('cuda' if torch.cuda.is_available() else 'cpu')

# IS Calculation
inception_score_value = inception_score(synthetic_loader, inception_model, 'cuda' if torch.cuda.is_available() else 'cpu')

print(f"Inception Score: {inception_score_value:.4f}")

