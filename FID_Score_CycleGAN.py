#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import torch
import numpy as np
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from scipy.linalg import sqrtm
from torch.nn.functional import adaptive_avg_pool2d
from PIL import Image


real_images_folder = "/media/orin/USBstore/CycleGAN_FID/Images"
synthetic_images_folder = "/media/orin/USBstore/CycleGAN_FID/CycleGAN_generated_images"


input_size = 512  


transform = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  
])


class SingleFolderDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image


real_dataset = SingleFolderDataset(real_images_folder, transform=transform)
synthetic_dataset = SingleFolderDataset(synthetic_images_folder, transform=transform)

real_loader = DataLoader(real_dataset, batch_size=32, shuffle=False, num_workers=4)
synthetic_loader = DataLoader(synthetic_dataset, batch_size=32, shuffle=False, num_workers=4)


inception_model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
inception_model.fc = torch.nn.Identity()  
inception_model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
inception_model.to(device)


def get_activations(dataloader, model, device):
    model.eval()
    activations = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            pred = model(batch)  
            
            if pred.dim() == 4:  
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))  
                pred = pred.squeeze(-1).squeeze(-1)  
            
            activations.append(pred.cpu().numpy())  
    
    activations = np.concatenate(activations, axis=0)
    return activations


def calculate_statistics(activations):
    mu = np.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)
    return mu, sigma


def calculate_fid(mu1, sigma1, mu2, sigma2, eps=1e-6):
    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1.dot(sigma2) + np.eye(sigma1.shape[0]) * eps, disp=False)
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid


real_activations = get_activations(real_loader, inception_model, device)
synthetic_activations = get_activations(synthetic_loader, inception_model, device)


mu_real, sigma_real = calculate_statistics(real_activations)
mu_synthetic, sigma_synthetic = calculate_statistics(synthetic_activations)

# FID score
fid_score = calculate_fid(mu_real, sigma_real, mu_synthetic, sigma_synthetic)

print(f"FID Score: {fid_score}")

