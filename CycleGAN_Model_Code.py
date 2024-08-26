#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import itertools
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt


torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.benchmark = True


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths for train dataset and output
train_data_dir = "/home/orin/SecurityData/train"
output_dir = "/home/orin/output3"
os.makedirs(output_dir, exist_ok=True)

# Hyperparameters
hidden_size = 128
image_channels = 3
image_size = 256  
num_epochs = 1000
batch_size = 10
learning_rate = 0.0002
betas = (0.5, 0.999)
lambda_cycle = 10  
lambda_identity = 0.5  
max_generated_images = 30  


transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

train_dataset = ImageDataset(train_data_dir, transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# Generator 
class Generator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, hidden_size, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hidden_size, hidden_size * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_size * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hidden_size * 2, hidden_size * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_size * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hidden_size * 4, hidden_size * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_size * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hidden_size * 8, hidden_size * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_size * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(hidden_size * 8, hidden_size * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_size * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(hidden_size * 8, hidden_size * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_size * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(hidden_size * 4, hidden_size * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_size * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(hidden_size * 2, hidden_size, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(True),

            nn.ConvTranspose2d(hidden_size, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

# Discriminator 
class Discriminator(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, hidden_size, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hidden_size, hidden_size * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_size * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hidden_size * 2, hidden_size * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_size * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hidden_size * 4, hidden_size * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_size * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hidden_size * 8, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


G_AB = Generator(image_channels, image_channels).to(device)
G_BA = Generator(image_channels, image_channels).to(device)
D_A = Discriminator(image_channels).to(device)
D_B = Discriminator(image_channels).to(device)

# MSE Loss
criterion_GAN = nn.MSELoss()
criterion_cycle = nn.L1Loss()
criterion_identity = nn.L1Loss()

optimizer_G = optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=learning_rate, betas=betas)
optimizer_D_A = optim.Adam(D_A.parameters(), lr=learning_rate, betas=betas)
optimizer_D_B = optim.Adam(D_B.parameters(), lr=learning_rate, betas=betas)


fake_A_buffer = []
fake_B_buffer = []

def update_image_buffer(buffer, images, buffer_size=50):
    if len(buffer) < buffer_size:
        buffer.append(images)
        return images
    if np.random.rand() > 0.5:
        index = np.random.randint(0, buffer_size)
        temp = buffer[index]
        buffer[index] = images
        return temp
    else:
        return images


g_losses = []
d_A_losses = []
d_B_losses = []

# Training
for epoch in range(num_epochs):
    for i, data in enumerate(train_loader):
        real_A = data.to(device)
        real_B = data.to(device)

        
        #  Discriminators training
       
        optimizer_D_A.zero_grad()
        optimizer_D_B.zero_grad()

       
        pred_real_A = D_A(real_A)
        pred_real_B = D_B(real_B)
        loss_D_real_A = criterion_GAN(pred_real_A, torch.ones_like(pred_real_A))
        loss_D_real_B = criterion_GAN(pred_real_B, torch.ones_like(pred_real_B))

        
        fake_B = G_AB(real_A)
        fake_A = G_BA(real_B)

        fake_B = update_image_buffer(fake_B_buffer, fake_B.detach())
        fake_A = update_image_buffer(fake_A_buffer, fake_A.detach())

        pred_fake_A = D_A(fake_A)
        pred_fake_B = D_B(fake_B)
        loss_D_fake_A = criterion_GAN(pred_fake_A, torch.zeros_like(pred_fake_A))
        loss_D_fake_B = criterion_GAN(pred_fake_B, torch.zeros_like(pred_fake_B))

        
        loss_D_A = (loss_D_real_A + loss_D_fake_A) * 0.5
        loss_D_B = (loss_D_real_B + loss_D_fake_B) * 0.5

        loss_D_A.backward()
        loss_D_B.backward()

        optimizer_D_A.step()
        optimizer_D_B.step()

        
        #  Generators training
        
        optimizer_G.zero_grad()

       
        loss_identity_A = criterion_identity(G_BA(real_A), real_A) * lambda_cycle * lambda_identity
        loss_identity_B = criterion_identity(G_AB(real_B), real_B) * lambda_cycle * lambda_identity

       
        pred_fake_B = D_B(fake_B)
        pred_fake_A = D_A(fake_A)
        loss_GAN_AB = criterion_GAN(pred_fake_B, torch.ones_like(pred_fake_B))
        loss_GAN_BA = criterion_GAN(pred_fake_A, torch.ones_like(pred_fake_A))

        
        loss_cycle_A = criterion_cycle(G_BA(fake_B), real_A) * lambda_cycle
        loss_cycle_B = criterion_cycle(G_AB(fake_A), real_B) * lambda_cycle

       
        loss_G = loss_GAN_AB + loss_GAN_BA + loss_cycle_A + loss_cycle_B + loss_identity_A + loss_identity_B

        loss_G.backward()
        optimizer_G.step()

        if (i + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], "
                  f"D_A_loss: {loss_D_A.item():.4f}, D_B_loss: {loss_D_B.item():.4f}, G_loss: {loss_G.item():.4f}")

    # Images are saving after particular epochs
    if (epoch + 1) % 50= 0:
        with torch.no_grad():
            image_count = 0
            for idx, img in enumerate(train_loader):
                img = img.to(device)
                synthetic_images = G_AB(img)
                synthetic_images = (synthetic_images + 1) / 2  
                for j in range(synthetic_images.size(0)):
                    if image_count < max_generated_images:
                        save_image(synthetic_images[j], os.path.join(output_dir, f'epoch_{epoch + 1}_img_{image_count + 1}.png'))
                        image_count += 1
                    else:
                        break
                if image_count >= max_generated_images:
                    break

    
    g_losses.append(loss_G.item())
    d_A_losses.append(loss_D_A.item())
    d_B_losses.append(loss_D_B.item())

# checkpoint
torch.save({
    'G_AB_state_dict': G_AB.state_dict(),
    'G_BA_state_dict': G_BA.state_dict(),
    'D_A_state_dict': D_A.state_dict(),
    'D_B_state_dict': D_B.state_dict(),
    'optimizer_G_state_dict': optimizer_G.state_dict(),
    'optimizer_D_A_state_dict': optimizer_D_A.state_dict(),
    'optimizer_D_B_state_dict': optimizer_D_B.state_dict(),
}, os.path.join(output_dir, 'final_checkpoint.pth'))


plt.figure(figsize=(20, 10))
plt.plot(np.arange(1, num_epochs + 1), g_losses, label='Generator Loss')
plt.plot(np.arange(1, num_epochs + 1), d_A_losses, label='Discriminator A Loss')
plt.plot(np.arange(1, num_epochs + 1), d_B_losses, label='Discriminator B Loss')
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Losses')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'training_curves.png'))
plt.show()

# Visualization
with torch.no_grad():
    image_count = 0
    for idx, img in enumerate(train_loader):
        img = img.to(device)
        synthetic_images = G_AB(img)
        synthetic_images = (synthetic_images + 1) / 2  
        for j in range(synthetic_images.size(0)):
            if image_count < max_generated_images:
                save_image(synthetic_images[j], os.path.join(output_dir, f'final_img_{image_count + 1}.png'))
                image_count += 1
            else:
                break
        if image_count >= max_generated_images:
            break

